import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class IDX3Reader {
    public static float[][][] readIDX3(String filePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filePath))) {
            // Read header information
            int magic = dis.readInt();  // Magic number (ignore)
            int numImages = dis.readInt();
            int numRows = dis.readInt();
            int numCols = dis.readInt();

            float[][][] images = new float[numImages][numRows][numCols];

            for (int i = 0; i < 2; i++) {
                for (int r = 0; r < numRows; r++) {
                    for (int c = 0; c < numCols; c++) {
                        images[i][r][c] = dis.readUnsignedByte() / 255.0f; // Normalize to [0,1]
                    }
                }
            }
            return images;
        }
    }
    public static float[] readIDX1(String filePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filePath))) {
            // Read magic number (ignore)
            int magic = dis.readInt();
            int numLabels = dis.readInt();

            float[] labels = new float[numLabels];

            for (int i = 0; i < numLabels; i++) {
                labels[i] = dis.readUnsignedByte() / 9.0f; // Normalize to [0,1] (optional)
            }

            return labels;
        }
    }

        public static float[] toArray(float[][] image){
        float[] array = new float[image.length*image[0].length];
        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[0].length; j++) {
                array[i*image[0].length+j] = image[i][j];
            }
        }
        return array;
    }
}
