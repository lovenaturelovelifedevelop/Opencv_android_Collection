package org.opencv.samples.opencv_mobilenet;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.res.AssetManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.util.Log;
import android.widget.Toast;

import org.json.JSONObject;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.samples.R;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import static android.content.pm.PackageManager.PERMISSION_GRANTED;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "OpenCV/Sample/MobileNet";

    private Net net;
    private CameraBridgeViewBase mOpenCvCameraView;
    private static final int NOT_NOTICE = 2;//如果勾选了不再询问
    private AlertDialog alertDialog;
    private AlertDialog mDialog;
    JSONObject classes = null;
    String sdcard =  Environment.getExternalStoragePublicDirectory("")+"";
    String modelDir = sdcard;

    static {
        System.loadLibrary("opencv_java4");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Set up camera listener
        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        myRequetPermission(this);
        mOpenCvCameraView.enableView();
        mOpenCvCameraView.setCameraPermissionGranted();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        try {
            // 从assets里移动模型文件到sdcard中
            String weights = getPath("ssd_mobilenet_v2_coco_2018_03_29.pb", this);
            String proto = getPath("ssd_mobilenet_v2_coco_2018_03_29.pbtxt", this);
            // label 文件
            String label = getPath("coco_labels.txt", this);

            File protoFile = new File(proto);
            if(!protoFile.exists()){
                Log.i(TAG, "File not found: " + proto);
            }
            File weightsFile = new File(proto);
            if(!weightsFile.exists()){
                Log.i(TAG, "File not found: " + weights);
            }

            File labelFile = new File(label);
            if(!labelFile.exists()){
                Log.i(TAG, "File not found: " + label);
            }
            // load a network
            // caffe模型
            //net = Dnn.readNetFromCaffe(proto, weights);

            // tensorflow模型
            net = Dnn.readNetFromTensorflow(weights, proto);
            Log.i(TAG, "Network loaded successfully");

            // 读取class
            String classString = readClasses(label);
           classes = new JSONObject(classString);
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStopped() {

    }

    /**
     * 相机接受帧，进行检测
     * @param inputFrame
     * @return
     */
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Log.i(TAG, "------onCameraFrame------");
        // get a new Frame
        Mat frame = inputFrame.rgba();
        try{
            final int IN_WIDTH = 300;
            final int IN_HEIGHT = 300;
            final float WH_RATIO = (float) IN_WIDTH / IN_HEIGHT;
            final double IN_SCALE_FACTOR = 1.0;
            final double MEAN_VAL = 127.5;
            final double THRESHOLD = 0.1;
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

            long startTime = System.currentTimeMillis();

            // Forward image through network
            Mat blob = Dnn.blobFromImage(frame, IN_SCALE_FACTOR,
                    new Size(IN_WIDTH, IN_HEIGHT));
            net.setInput(blob);
            Mat detections = net.forward();

            int cols = frame.cols();
            int rows = frame.rows();
            detections = detections.reshape(1, (int)detections.total() / 7);
            long endTime = System.currentTimeMillis();
            Log.i(TAG, "------detect time(ms) ------：" + (endTime - startTime));
            for (int i = 0; i < detections.rows(); ++i) {
                double confidence = detections.get(i, 2)[0];
                Log.i(TAG, "------confidence------：" + confidence);
                if (confidence > THRESHOLD) {
                    int classId = (int)detections.get(i, 1)[0];
                    int left   = (int)(detections.get(i, 3)[0] * cols);
                    int top    = (int)(detections.get(i, 4)[0] * rows);
                    int right  = (int)(detections.get(i, 5)[0] * cols);
                    int bottom = (int)(detections.get(i, 6)[0] * rows);
                    // Draw rectangle around detected object.
                    Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                            new Scalar(0, 255, 0));
                    String label = classId + "_" + classes.getString(classId+"") + ": " + confidence;
                    int[] baseLine = new int[1];
                    Size labelSize = Imgproc.getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
                    // Draw background for label.
                    Imgproc.rectangle(frame, new Point(left, top - labelSize.height),
                            new Point(left + labelSize.width, top + baseLine[0]),
                            new Scalar(255, 255, 255), Core.FILLED);
                    // Write class name and confidence.
                    Imgproc.putText(frame, label, new Point(left, top),
                            FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0));
                }
            }
        }catch (Exception e){
            e.printStackTrace();
        }

        return frame;
    }

    /**
     * 确认权限
     * @param activity
     * @return
     */
    public void myRequetPermission(Activity activity) {
        // camera 权限，读写权限
        //判断当前是否已经有某个权限
        if (ContextCompat.checkSelfPermission(activity,android.Manifest.permission.CAMERA)!= PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(activity,android.Manifest.permission.WRITE_EXTERNAL_STORAGE)!= PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(activity, Manifest.permission.READ_EXTERNAL_STORAGE)!= PERMISSION_GRANTED) {
            String[] permissions = new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE};
            //申请权限，字符串数组内是一个或多个要申请的权限，1是申请权限结果的返回参数，在onRequestPermissionsResult可以得知申请结果
            ActivityCompat.requestPermissions(
                    activity,
                    permissions,
                    1);
        } else {
            Toast.makeText(activity,"已经申请CAMERA, READ , WRITE权限!", Toast.LENGTH_LONG).show();
        }
    }

    // 用户操作后的回调
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == 1) {
            for (int i = 0; i < permissions.length; i++) {
                if (grantResults[i] == PERMISSION_GRANTED) {//选择了“始终允许”
                    Toast.makeText(this, "" + "权限" + permissions[i] + "申请成功", Toast.LENGTH_SHORT).show();
                } else {
                    if (!ActivityCompat.shouldShowRequestPermissionRationale(this, permissions[i])){//用户选择了禁止不再询问

                        AlertDialog.Builder builder = new AlertDialog.Builder(this);
                        builder.setTitle("permission")
                                .setMessage("点击允许才可以使用我们的app哦")
                                .setPositiveButton("去允许", new DialogInterface.OnClickListener() {
                                    public void onClick(DialogInterface dialog, int id) {
                                        if (mDialog != null && mDialog.isShowing()) {
                                            mDialog.dismiss();
                                        }
                                        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                                        Uri uri = Uri.fromParts("package", getPackageName(), null);//注意就是"package",不用改成自己的包名
                                        intent.setData(uri);
                                        startActivityForResult(intent, NOT_NOTICE);
                                    }
                                });
                        mDialog = builder.create();
                        mDialog.setCanceledOnTouchOutside(false);
                        mDialog.show();



                    }else {//选择禁止
                        AlertDialog.Builder builder = new AlertDialog.Builder(this);
                        builder.setTitle("permission")
                                .setMessage("点击允许才可以使用我们的app哦")
                                .setPositiveButton("去允许", new DialogInterface.OnClickListener() {
                                    public void onClick(DialogInterface dialog, int id) {
                                        if (alertDialog != null && alertDialog.isShowing()) {
                                            alertDialog.dismiss();
                                        }
                                        ActivityCompat.requestPermissions(MainActivity.this,
                                                new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
                                    }
                                });
                        alertDialog = builder.create();
                        alertDialog.setCanceledOnTouchOutside(false);
                        alertDialog.show();
                    }

                }
            }
        }
    }

    private String getPath(String file, Context context){
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(new File(modelDir), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            ex.printStackTrace();
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

    private String readClasses(String filePath){
        StringBuilder builder = new StringBuilder();
        InputStream inputStream = null;
        BufferedReader br = null;
        String content = "";
        try {
            inputStream = new FileInputStream(new File(filePath));
            br = new BufferedReader(new InputStreamReader(inputStream));
            char[] chars = new char[1024];
            boolean var7 = true;

            int length;
            while ((length = br.read(chars)) != -1) {
                builder.append(chars, 0, length);
            }

            content = builder.toString();

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (br != null) {
                    br.close();
                }
                if (inputStream != null) {
                    inputStream.close();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return content;
    }

}
