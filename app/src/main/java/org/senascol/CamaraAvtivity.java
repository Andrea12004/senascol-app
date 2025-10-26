package org.senascol.senascol;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.google.common.util.concurrent.ListenableFuture;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CameraActivity extends AppCompatActivity {
    
    private PreviewView previewView;
    private TextView tvStatus;
    private TextView tvResult;
    private ProgressBar progressBar;
    private Button btnStart, btnStop, btnClear;
    
    private PyObject translator;
    private ExecutorService cameraExecutor;
    private Camera camera;
    private boolean isProcessing = false;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        
        // Inicializar vistas
        previewView = findViewById(R.id.previewView);
        tvStatus = findViewById(R.id.tvStatus);
        tvResult = findViewById(R.id.tvResult);
        progressBar = findViewById(R.id.progressBar);
        btnStart = findViewById(R.id.btnStart);
        btnStop = findViewById(R.id.btnStop);
        btnClear = findViewById(R.id.btnClear);
        
        // Inicializar traductor Python
        Python py = Python.getInstance();
        PyObject module = py.getModule("traductor");
        translator = module.callAttr("SignLanguageTranslator");
        
        // Executor para cámara
        cameraExecutor = Executors.newSingleThreadExecutor();
        
        // Botones
        btnStart.setOnClickListener(v -> {
            btnStart.setEnabled(false);
            btnStop.setEnabled(true);
            isProcessing = true;
            startCamera();
            tvStatus.setText("✅ Cámara activa");
        });
        
        btnStop.setOnClickListener(v -> {
            btnStart.setEnabled(true);
            btnStop.setEnabled(false);
            isProcessing = false;
            tvStatus.setText("⏹️ Cámara detenida");
        });
        
        btnClear.setOnClickListener(v -> {
            translator.callAttr("reset");
            tvResult.setText("La traducción aparecerá aquí...");
            tvStatus.setText("🧹 Texto limpiado");
        });
        
        // Botón volver
        findViewById(R.id.btnBack).setOnClickListener(v -> finish());
    }
    
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);
        
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        // Preview
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        
        // Análisis de imagen
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        
        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeImage);
        
        // Cámara frontal
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build();
        
        // Bind al lifecycle
        cameraProvider.unbindAll();
        camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalysis);
    }
    
    private void analyzeImage(@NonNull ImageProxy image) {
        if (!isProcessing) {
            image.close();
            return;
        }
        
        try {
            // Convertir ImageProxy a Bitmap
            Bitmap bitmap = imageProxyToBitmap(image);
            
            // Procesar con Python
            PyObject result = translator.callAttr("process_frame_android", bitmap);
            
            // Actualizar UI
            runOnUiThread(() -> updateUI(result));
            
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            image.close();
        }
    }
    
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();
        
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        
        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);
        
        android.graphics.YuvImage yuvImage = new android.graphics.YuvImage(
                nv21, android.graphics.ImageFormat.NV21,
                image.getWidth(), image.getHeight(), null);
        
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        yuvImage.compressToJpeg(
                new android.graphics.Rect(0, 0, image.getWidth(), image.getHeight()),
                100, out);
        
        byte[] imageBytes = out.toByteArray();
        Bitmap bitmap = android.graphics.BitmapFactory.decodeByteArray(
                imageBytes, 0, imageBytes.length);
        
        // Rotar si es necesario
        Matrix matrix = new Matrix();
        matrix.postRotate(image.getImageInfo().getRotationDegrees());
        
        return Bitmap.createBitmap(bitmap, 0, 0,
                bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }
    
    private void updateUI(PyObject result) {
        String status = result.get("status").toString();
        
        switch (status) {
            case "success":
                String prediction = result.get("prediction").toString();
                String type = result.get("type").toString();
                
                if (type.equals("static")) {
                    tvStatus.setText("🔤 LETRA: " + prediction);
                } else {
                    tvStatus.setText("👋 PALABRA: " + prediction);
                }
                
                // Actualizar resultado
                PyObject sentence = translator.callAttr("get_sentence");
                tvResult.setText(sentence.toString());
                progressBar.setProgress(0);
                break;
                
            case "accumulating":
                String counter = result.get("counter").toString();
                int progress = result.get("progress").toJava(Integer.class);
                tvStatus.setText("⏳ Detectando... " + counter);
                progressBar.setProgress(progress);
                break;
                
            case "waiting":
                tvStatus.setText("✋ Muestra las manos");
                progressBar.setProgress(0);
                break;
                
            case "detecting":
                tvStatus.setText("👋 Detectando...");
                break;
                
            case "error":
                String error = result.get("message").toString();
                tvStatus.setText("❌ " + error);
                break;
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }
}