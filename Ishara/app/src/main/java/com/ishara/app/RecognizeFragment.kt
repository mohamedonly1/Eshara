package com.ishara.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.GestureDetector
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

class RecognizeFragment : Fragment() {

    companion object {
        private const val TAG = "ISHARA"
        private const val CONFIRMATION_WINDOW = 25
        private const val CONFIRMATION_THRESHOLD = 0.85f
        private const val COOLDOWN_FRAMES = 40
    }

    private lateinit var previewView: PreviewView
    private lateinit var handOverlay: HandOverlayView
    private lateinit var currentLetterPill: TextView
    private lateinit var confidenceText: TextView
    private lateinit var confirmedText: TextView
    private lateinit var clearButton: MaterialButton
    private lateinit var switchCameraButton: ImageButton

    private var handLandmarker: HandLandmarker? = null
    private var tfliteInterpreter: Interpreter? = null
    private var labelsMap: Map<Int, String> = emptyMap()

    private val predictionHistory = ArrayDeque<Int>()
    private var cooldownCounter = 0
    private val confirmedLetters = StringBuilder()

    // Default to back camera
    private var isBackCamera = true

    private lateinit var cameraExecutor: ExecutorService

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                startCamera()
            } else {
                Toast.makeText(
                    requireContext(),
                    getString(R.string.camera_permission_denied),
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_recognize, container, false)

        previewView = view.findViewById(R.id.previewView)
        handOverlay = view.findViewById(R.id.handOverlay)
        currentLetterPill = view.findViewById(R.id.currentLetterPill)
        confidenceText = view.findViewById(R.id.confidenceText)
        confirmedText = view.findViewById(R.id.confirmedText)
        clearButton = view.findViewById(R.id.clearButton)
        switchCameraButton = view.findViewById(R.id.switchCameraButton)

        clearButton.setOnClickListener {
            confirmedLetters.clear()
            confirmedText.text = ""
            predictionHistory.clear()
            cooldownCounter = 0
            currentLetterPill.text = "ـ"
            confidenceText.text = "0%"
        }

        switchCameraButton.setOnClickListener {
            switchCamera()
        }

        // Double-tap on preview to switch camera
        val doubleTapDetector = GestureDetector(requireContext(),
            object : GestureDetector.SimpleOnGestureListener() {
                override fun onDoubleTap(e: MotionEvent): Boolean {
                    switchCamera()
                    return true
                }
            })
        previewView.setOnTouchListener { _, event ->
            doubleTapDetector.onTouchEvent(event)
            false
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        loadLabels()
        setupHandLandmarker()
        setupTFLite()

        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        if (ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // ============================================================
    // Camera switching
    // ============================================================
    private fun switchCamera() {
        isBackCamera = !isBackCamera
        // Update overlay mirroring: front camera = mirror, back camera = no mirror
        handOverlay.setMirror(!isBackCamera)
        startCamera()
    }

    // ============================================================
    // Labels loading — matches server.py exactly:
    //   labels_dict = {}
    //   with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    //       for row in csv.reader(f):
    //           labels_dict[int(row[0])] = row[1]
    // ============================================================
    private fun loadLabels() {
        try {
            val inputStream = requireContext().assets.open("arabic_labels.csv")
            val reader = BufferedReader(InputStreamReader(inputStream, Charsets.UTF_8))
            val map = mutableMapOf<Int, String>()

            reader.forEachLine { line ->
                val parts = line.split(",", limit = 2)
                if (parts.size == 2) {
                    val index = parts[0].trim().toIntOrNull()
                    val letter = parts[1].trim()
                    if (index != null && letter.isNotEmpty()) {
                        map[index] = letter
                    }
                }
            }

            reader.close()
            labelsMap = map
            Log.d(TAG, "Loaded ${labelsMap.size} labels")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load labels: ${e.message}")
        }
    }

    // ============================================================
    // MediaPipe HandLandmarker setup
    // ============================================================
    private fun setupHandLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.7f)
                .setMinHandPresenceConfidence(0.7f)
                .setMinTrackingConfidence(0.5f)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(requireContext(), options)
            Log.d(TAG, "HandLandmarker initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize HandLandmarker: ${e.message}")
        }
    }

    // ============================================================
    // TFLite Interpreter setup
    // ============================================================
    private fun setupTFLite() {
        try {
            val model = loadModelFile("arabic_sign_model.tflite")
            tfliteInterpreter = Interpreter(model)

            // Log model input/output shapes for debugging
            val inputShape = tfliteInterpreter!!.getInputTensor(0).shape()
            val outputShape = tfliteInterpreter!!.getOutputTensor(0).shape()
            Log.d(TAG, "TFLite input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "TFLite output shape: ${outputShape.contentToString()}")
            Log.d(TAG, "TFLite interpreter initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model: ${e.message}")
        }
    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        val assetFileDescriptor = requireContext().assets.openFd(filename)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // ============================================================
    // CameraX setup
    // ============================================================
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val cameraSelector = if (isBackCamera)
                CameraSelector.DEFAULT_BACK_CAMERA
            else
                CameraSelector.DEFAULT_FRONT_CAMERA

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                processFrame(imageProxy)
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    viewLifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
                Log.d(TAG, "Camera bound: ${if (isBackCamera) "BACK" else "FRONT"}")
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    // ============================================================
    // Normalize landmarks — corrects for Android portrait aspect ratio 
    // to match the 640x480 landscape aspect ratio used during training.
    // ============================================================
    private fun normalizeLandmarks(
        landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int
    ): FloatArray {
        // At this point, the coordinates are guaranteed to be from an UPRIGHT image.
        // e.g. imageWidth = 480, imageHeight = 640.
        // We scale them to match the 640x480 webcam aspect ratio.
        val scaleX = imageWidth / 640f
        val scaleY = imageHeight / 480f

        val points = landmarks.map { Pair(it.x() * scaleX, it.y() * scaleY) }

        val baseX = points[0].first
        val baseY = points[0].second
        val rel = points.flatMap {
            listOf(it.first - baseX, it.second - baseY)
        }
        val maxVal = rel.maxOfOrNull { abs(it) }
            ?.takeIf { it != 0f } ?: 1f
        return rel.map { it / maxVal }.toFloatArray()
    }

    // ============================================================
    // Frame processing pipeline
    // ============================================================
    private fun processFrame(imageProxy: ImageProxy) {
        val landmarker = handLandmarker ?: run {
            imageProxy.close()
            return
        }
        val interpreter = tfliteInterpreter ?: run {
            imageProxy.close()
            return
        }

        try {
            var bitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            // Foolproof approach: Guarantee the bitmap is upright BEFORE giving it to MediaPipe.
            // If the bitmap is landscape (width > height), we manually rotate it.
            if (bitmap.width > bitmap.height) {
                val matrix = android.graphics.Matrix()
                matrix.postRotate(rotationDegrees.toFloat())
                bitmap = android.graphics.Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                )
            }

            val processedWidth = bitmap.width
            val processedHeight = bitmap.height

            // Now the image is guaranteed to be upright, so MediaPipe needs 0 rotation.
            val imageProcessingOptions = ImageProcessingOptions.builder()
                .setRotationDegrees(0)
                .build()

            val mpImage = BitmapImageBuilder(bitmap).build()
            val result: HandLandmarkerResult = landmarker.detect(mpImage, imageProcessingOptions)

            if (result.landmarks().isEmpty()) {
                activity?.runOnUiThread {
                    handOverlay.clear()
                    currentLetterPill.text = "ـ"
                    confidenceText.text = getString(R.string.no_hand_detected)
                }
                imageProxy.close()
                return
            }

            // Draw hand landmarks on overlay
            activity?.runOnUiThread {
                handOverlay.setResults(result, processedWidth, processedHeight, isBackCamera)
            }

            val landmarks = result.landmarks()[0]

            // Normalize landmarks with aspect ratio correction
            val normalized = normalizeLandmarks(landmarks, processedWidth, processedHeight)

            // Run TFLite inference
            val inputBuffer = ByteBuffer.allocateDirect(42 * 4).apply {
                order(ByteOrder.nativeOrder())
                for (v in normalized) putFloat(v)
            }

            val outputArray = Array(1) { FloatArray(labelsMap.size) }
            interpreter.run(inputBuffer, outputArray)

            val probs = outputArray[0]

            // Find top-3 predictions for debug logging
            data class Prediction(val idx: Int, val prob: Float)
            val top3 = probs.mapIndexed { i, p -> Prediction(i, p) }
                .sortedByDescending { it.prob }
                .take(3)

            val maxIdx = top3[0].idx
            val maxProb = top3[0].prob

            val predictedLetter = labelsMap[maxIdx] ?: "?"
            val confidencePct = (maxProb * 100).toInt()

            // Debug log top-3
            val topStr = top3.joinToString(", ") { "${labelsMap[it.idx] ?: "?"} ${(it.prob * 100).toInt()}%" }
            Log.d(TAG, "Top: $topStr")

            // ============================================================
            // Confirmation system: 85% of last 25 frames must agree
            // 40-frame cooldown after each confirmed letter
            // ============================================================
            if (cooldownCounter > 0) {
                cooldownCounter--
            } else {
                predictionHistory.addLast(maxIdx)
                if (predictionHistory.size > CONFIRMATION_WINDOW) {
                    predictionHistory.removeFirst()
                }

                if (predictionHistory.size == CONFIRMATION_WINDOW) {
                    val counts = mutableMapOf<Int, Int>()
                    for (p in predictionHistory) {
                        counts[p] = (counts[p] ?: 0) + 1
                    }
                    val bestEntry = counts.maxByOrNull { it.value }
                    if (bestEntry != null) {
                        val bestIdx = bestEntry.key
                        val bestCount = bestEntry.value
                        val ratio = bestCount.toFloat() / CONFIRMATION_WINDOW

                        if (ratio >= CONFIRMATION_THRESHOLD) {
                            val confirmedLetter = labelsMap[bestIdx] ?: "?"
                            confirmedLetters.append(confirmedLetter)
                            predictionHistory.clear()
                            cooldownCounter = COOLDOWN_FRAMES
                            Log.d(TAG, "CONFIRMED: $confirmedLetter (${(ratio * 100).toInt()}%)")
                        }
                    }
                }
            }

            activity?.runOnUiThread {
                currentLetterPill.text = predictedLetter
                confidenceText.text = "$confidencePct%"
                confirmedText.text = confirmedLetters.toString()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        cameraExecutor.shutdown()
        handLandmarker?.close()
        tfliteInterpreter?.close()
    }
}
