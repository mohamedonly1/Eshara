package com.ishara.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.ConsoleMessage
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment

class TranslateFragment : Fragment() {

    companion object {
        private const val TAG = "IsharaWebView"
    }

    private var webView: WebView? = null
    private var speechRecognizer: SpeechRecognizer? = null

    private val requestAudioPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                startListening()
            } else {
                Toast.makeText(requireContext(), "يجب منح إذن الميكروفون", Toast.LENGTH_SHORT).show()
            }
        }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_translate, container, false)
        webView = view.findViewById(R.id.webView)

        webView?.apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            settings.allowFileAccessFromFileURLs = true
            settings.allowUniversalAccessFromFileURLs = true
            settings.mediaPlaybackRequiresUserGesture = false

            addJavascriptInterface(SpeechBridge(), "AndroidSpeech")

            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(consoleMessage: ConsoleMessage?): Boolean {
                    consoleMessage?.let {
                        Log.d(TAG, "${it.message()} -- From line ${it.lineNumber()} of ${it.sourceId()}")
                    }
                    return true
                }
            }

            loadUrl("file:///android_asset/translate.html")
        }

        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Inject JS override for the mic button after page loads
        webView?.webViewClient = object : android.webkit.WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                // Override the mic button to call Android's SpeechRecognizer
                val js = """
                    (function() {
                        var micBtn = document.getElementById('micBtn');
                        if (micBtn) {
                            micBtn.onclick = function() {
                                if (typeof AndroidSpeech !== 'undefined') {
                                    AndroidSpeech.startListening();
                                }
                            };
                        }
                    })();
                """.trimIndent()
                view?.evaluateJavascript(js, null)
            }
        }
    }

    inner class SpeechBridge {
        @JavascriptInterface
        fun startListening() {
            activity?.runOnUiThread {
                if (ContextCompat.checkSelfPermission(
                        requireContext(), Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED
                ) {
                    startListeningInternal()
                } else {
                    requestAudioPermission.launch(Manifest.permission.RECORD_AUDIO)
                }
            }
        }
    }

    private fun startListening() {
        startListeningInternal()
    }

    private fun startListeningInternal() {
        if (!SpeechRecognizer.isRecognitionAvailable(requireContext())) {
            Toast.makeText(requireContext(), "التعرف على الكلام غير متاح", Toast.LENGTH_SHORT).show()
            return
        }

        speechRecognizer?.destroy()
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(requireContext())

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ar")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }

        // Update mic button style to "listening"
        webView?.evaluateJavascript(
            "document.getElementById('micBtn').classList.add('listening');", null
        )

        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {
                activity?.runOnUiThread {
                    webView?.evaluateJavascript(
                        "document.getElementById('micBtn').classList.remove('listening');", null
                    )
                }
            }

            override fun onError(error: Int) {
                Log.e(TAG, "Speech recognition error: $error")
                activity?.runOnUiThread {
                    webView?.evaluateJavascript(
                        "document.getElementById('micBtn').classList.remove('listening');", null
                    )
                }
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = matches?.firstOrNull() ?: return
                Log.d(TAG, "Speech result: $text")
                activity?.runOnUiThread {
                    // Escape single quotes for JS
                    val safeText = text.replace("'", "\\'").replace("\n", " ")
                    webView?.evaluateJavascript(
                        "document.getElementById('textInput').value = '$safeText';", null
                    )
                    webView?.evaluateJavascript(
                        "document.getElementById('micBtn').classList.remove('listening');", null
                    )
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        speechRecognizer?.startListening(intent)
    }

    override fun onDestroyView() {
        speechRecognizer?.destroy()
        speechRecognizer = null
        webView?.destroy()
        webView = null
        super.onDestroyView()
    }
}
