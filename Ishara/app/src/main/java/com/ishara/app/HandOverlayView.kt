package com.ishara.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

class HandOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var result: HandLandmarkerResult? = null
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var isBackCamera: Boolean = true

    init {
        setBackgroundColor(Color.TRANSPARENT)
    }

    private val pointPaint = Paint().apply {
        color = Color.parseColor("#00d4aa")
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val linePaint = Paint().apply {
        color = Color.parseColor("#7c3aed")
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    // MediaPipe hand landmark connections
    private val connections = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 4,       // thumb
        0 to 5, 5 to 6, 6 to 7, 7 to 8,       // index
        0 to 9, 9 to 10, 10 to 11, 11 to 12,  // middle
        0 to 13, 13 to 14, 14 to 15, 15 to 16, // ring
        0 to 17, 17 to 18, 18 to 19, 19 to 20, // pinky
        5 to 9, 9 to 13, 13 to 17              // palm
    )

    fun setMirror(mirror: Boolean) {
        // No longer using mirrorX directly, using isBackCamera logic
    }

    fun setResults(
        handLandmarkerResult: HandLandmarkerResult?,
        imgWidth: Int,
        imgHeight: Int,
        isBack: Boolean
    ) {
        result = handLandmarkerResult
        imageWidth = imgWidth
        imageHeight = imgHeight
        isBackCamera = isBack
        invalidate()
    }

    fun clear() {
        result = null
        invalidate()
    }

    /**
     * Maps raw sensor coordinates to upright view coordinates.
     * MediaPipe returns coordinates in the original unrotated sensor space.
     */
    private fun mapLandmark(normalizedX: Float, normalizedY: Float): Pair<Float, Float> {
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        // Coordinates are guaranteed to be upright because we manually rotated the bitmap
        // before passing it to MediaPipe.
        var screenX = normalizedX * viewWidth
        val screenY = normalizedY * viewHeight

        // Mirror X for front camera (since PreviewView acts as a mirror)
        if (!isBackCamera) {
            screenX = viewWidth - screenX
        }

        return Pair(screenX, screenY)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val res = result ?: return
        if (res.landmarks().isEmpty()) return

        val landmarks = res.landmarks()[0]

        // Draw connections
        for ((start, end) in connections) {
            if (start < landmarks.size && end < landmarks.size) {
                val startLm = landmarks[start]
                val endLm = landmarks[end]
                
                val (startX, startY) = mapLandmark(startLm.x(), startLm.y())
                val (endX, endY) = mapLandmark(endLm.x(), endLm.y())
                
                canvas.drawLine(startX, startY, endX, endY, linePaint)
            }
        }

        // Draw landmark points
        for (lm in landmarks) {
            val (x, y) = mapLandmark(lm.x(), lm.y())
            canvas.drawCircle(x, y, 8f, pointPaint)
        }
    }
}
