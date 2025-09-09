/**
 * Converts a 28x28 MNIST image into an array of grayscale pixel values
 */
import java.io.FileInputStream;
import java.io.IOException;

public class MnistImageProcessor {
    public static float[] readMnistImage(String filePath) throws IOException {
        float[] pixels = new float[28 * 28];

        try (FileInputStream fis = new FileInputStream(filePath)) {
            byte[] buffer = new byte[28 * 28];
            fis.read(buffer);

            for (int i = 0; i < buffer.length; i++) {
                // Get the unsigned byte value (0-255)
                int value = buffer[i] & 0xFF;

                // Invert the value since in image files typically:
                // - 0 = black/background
                // - 255 = white/foreground (digit)
                // But in CSV:
                // - 0 = background
                // - Positive values = digit
                pixels[i] = 255 - value;
            }
        }

        return pixels;
    }

}