package com.example.alberto.glide_demo;

import android.app.Activity;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemClickListener;
import android.widget.Gallery;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 *
 * @author danielme.com
 *
 */
public class MainActivity extends Activity {

    ImageView imagenSeleccionada;
    Gallery gallery;



    private static String MODEL_FILE = "file:///android_asset/optimized_graph.pb";
    private static final String INPUT_NODE = "x";
    private static final String OUTPUT_NODE = "y_pred";
    private static final int[] INPUT_SIZE = {1,128,128,3};

    private float[] floatValues = new float[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * 3];;
    private int imageMean = 128;
    private float imageStd = 64.0f;
    private static final int DIM_BATCH_SIZE = 32;

    private static final int DIM_PIXEL_SIZE = 3;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    static final int DIM_IMG_SIZE_X = 128;
    static final int DIM_IMG_SIZE_Y = 128;
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
    private ByteBuffer imgData = null;

    /** An array to hold inference results, to be feed into Tensorflow Lite as outputs. */
    private float[][] labelProbArray = null;
    /** multi-stage low pass filter * */
    private float[][] filterLabelProbArray = null;

    private static final int FILTER_STAGES = 2;
    private static final float FILTER_FACTOR = 0.4f;

    private String[] labelList  = new String[]{"caricatura", "documental"};//, "futbol", "guerra", "noticia", "novela"};

    private TensorFlowInferenceInterface inferenceInterface;

    static {
        System.loadLibrary("tensorflow_inference");
    }


    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);

        imagenSeleccionada = (ImageView) findViewById(R.id.seleccionada);

        final String[] imagenes = datos();

        gallery = (Gallery) findViewById(R.id.gallery);
        gallery.setAdapter(new GalleryAdapter(this,imagenes));
        //al seleccionar una imagen, la mostramos en el centro de la pantalla a mayor tamaño

        //con este listener, sólo se mostrarían las imágenes sobre las que se pulsa
        gallery.setOnItemClickListener(new OnItemClickListener()
        {
            public void onItemClick(AdapterView parent, View v, int position, long id)
            {
                imagenSeleccionada.setImageBitmap(BitmapUtils.decodeSampledBitmapFromResource(getResources(), imagenes[position], 300, 0));


                final File image = new File(imagenes[position]);
                BitmapFactory.Options bmOptions = new BitmapFactory.Options();

                Bitmap finalBitmap;
                finalBitmap = BitmapFactory.decodeFile(image.getAbsolutePath(),bmOptions);


                finalBitmap = Bitmap.createScaledBitmap(finalBitmap,DIM_IMG_SIZE_X,DIM_IMG_SIZE_Y,true);

                imgData =
                        ByteBuffer.allocateDirect(
                                DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
                imgData.order(ByteOrder.nativeOrder());
                labelProbArray = new float[1][labelList.length];
                filterLabelProbArray = new float[FILTER_STAGES][labelList.length];
                imgData.rewind();
                finalBitmap.getPixels(intValues, 0, finalBitmap.getWidth(), 0, 0, finalBitmap.getWidth(), finalBitmap.getHeight());
                // Convert the image to floating point.
                int pixel = 0;
                for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
                    for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                        final int val = intValues[pixel++];
                        floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
                        floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
                        floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
                    }
                }

                inferenceInterface.fillNodeFloat(INPUT_NODE,INPUT_SIZE,floatValues);



                inferenceInterface.runInference(new String[] {OUTPUT_NODE});

                float[] resu = {0,0};
                inferenceInterface.readNodeFloat(OUTPUT_NODE, resu);
                int index =  posicion(resu);

                final TextView salida = (TextView) findViewById(R.id.salid);
                switch (index) {
                    case 0:
                        salida.setText("0 : "+Float.toString(resu[0])+"\n1 : "+Float.toString(resu[1])+"\n"+index+" : Perro/Dog");
                        break;
                    case 1:
                        salida.setText("0 : "+Float.toString(resu[0])+"\n1 : "+Float.toString(resu[1])+"\n"+index+" : Gato/Cat");
                }
            }

        });

        //con este otro listener se mostraría la imagen seleccionada en la galería, esto es, la que se encuentre en el centro en cada momento
//      gallery.setOnItemSelectedListener(new OnItemSelectedListener() {
//
//          @Override
//          public void onItemSelected(AdapterView parent, View v, int position, long id)
//          {
//              imagenSeleccionada.setImageBitmap(BitmapUtils.decodeSampledBitmapFromResource(getResources(), imagenes[position], 400, 0));
//          }
//
//          @Override
//          public void onNothingSelected(AdapterView<?> arg0)
//          {
//              // TODO Auto-generated method stub
//
//          }
//      });

    }

    private String[] datos(){
        int count = 0;
        String[] projection = {

                MediaStore.Files.FileColumns._ID,//0
                MediaStore.Files.FileColumns.DATA,//1
                MediaStore.Files.FileColumns.DATE_ADDED,//2
                MediaStore.Files.FileColumns.MEDIA_TYPE,//3
                MediaStore.Files.FileColumns.MIME_TYPE,//4
                MediaStore.Files.FileColumns.TITLE,//5
                MediaStore.Files.FileColumns.SIZE,//6


        };

        String selection = MediaStore.Files.FileColumns.MEDIA_TYPE + "="
                + MediaStore.Files.FileColumns.MEDIA_TYPE_IMAGE;

        Uri queryUri = MediaStore.Files.getContentUri("external");

        ;
        Cursor cursor;
        cursor = managedQuery(queryUri,
                projection, // Which columns to return
                selection, //  MEDIA_DATA + " like ? ",       // WHERE clause; which rows to return (all rows)
                null,//    new String[] {"%"+PATH+"%"}, //  new String[] {"%sdcard%"},       // WHERE clause selection arguments (none)
                null); // Order-by clause (ascending by name)
        count = cursor.getCount();
        //ContentResolver mContentResolver = mContext.getContentResolver();
        // cursor = mContentResolver.query(queryUri, columns, MediaStore.Images.Media.DATA + " like ? ",new String[] {"%/YourFolderName/%"}, null);

        int k = 0;
        Integer[] id = new Integer[count];
        String[] titulo = new String[count];
        String[] ruta = new String[count];
        String[] nomb = new String[count];
        String[] tipo = new String[count];
        Bitmap[] imagen = new Bitmap[count];
        String[] fecha = new String[count];
        String[] tamano = new String[count];

        OutputStream outStream = null;
        String nombre;

        if (cursor.moveToFirst()) {
            do {
                id[k] = cursor.getInt(0);
                titulo[k] = cursor.getString(5);
                nombre = titulo[k] + ".png";
                tipo[k] = cursor.getString(4);
                ruta[k] = cursor.getString(1);
                titulo[k] = titulo[k] + "." + tipo[k].substring(6);
                tamano[k] = cursor.getString(6);

                Log.e("fotos",""+id[k]);
/*
                File file = new File(PATH, nombre);

                if (file.exists()) {
                    FileInputStream fis = null;
                    try {
                        fis = new FileInputStream(file);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                    imagen[k] = BitmapFactory.decodeStream(fis);
                } else {
                    try {
                        outStream = new FileOutputStream(file);
                        imagen[k] = ThumbnailUtils.createVideoThumbnail(ruta[k], MediaStore.Video.Thumbnails.MICRO_KIND);
                        imagen[k].compress(Bitmap.CompressFormat.PNG, 90, outStream);
                        outStream.flush();
                        outStream.close();
                    } catch (FileNotFoundException e) {
                        Log.d("", "File not found: " + e.getMessage());
                    } catch (IOException e) {
                        Log.d("", "Error accessing file: " + e.getMessage());
                    }
                }*/
                k++;
            } while (cursor.moveToNext());
        }
        return ruta;
    }

    public int posicion(float[] labels)
    {
        double numeromayor = 0.0;
        int posicion = 0;
        for(int i=0; i<labels.length; i++){
            // System.out.println(nombres[i] + " " + sueldos[i]);
            if(labels[i]>numeromayor){ //
                numeromayor = labels[i];
                posicion = i;
            }
        }

        System.out.println(numeromayor);
        return posicion;
    }

}