package org.senascol.senascol;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class MainActivity extends AppCompatActivity {
    
    private static final int CAMERA_PERMISSION_CODE = 100;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Inicializar Python/Chaquopy
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        
        // Botones
        Button btnSenasATexto = findViewById(R.id.btnSenasATexto);
        Button btnTextoASenas = findViewById(R.id.btnTextoASenas);
        
        // Señas a Texto (requiere cámara)
        btnSenasATexto.setOnClickListener(v -> {
            if (checkCameraPermission()) {
                startActivity(new Intent(MainActivity.this, CameraActivity.class));
            } else {
                requestCameraPermission();
            }
        });
        
        // Texto a Señas (por ahora solo muestra mensaje)
        btnTextoASenas.setOnClickListener(v -> {
            Toast.makeText(this, "Función en desarrollo", Toast.LENGTH_SHORT).show();
        });
    }
    
    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }
    
    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_CODE);
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startActivity(new Intent(MainActivity.this, CameraActivity.class));
            } else {
                Toast.makeText(this, "Se necesita permiso de cámara", Toast.LENGTH_SHORT).show();
            }
        }
    }
}