import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;
    //----------------//
    //File Management//
    //--------------//

public class FileManager {
    //clears file
    public static void clearCSV(String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //Reads a single line in a CSV file
    public static String readRow(String filename, int rowNumber){

        try(BufferedReader reader = new BufferedReader(new FileReader(filename))){
            String line;
            for(int i = 0; i < rowNumber-1; i++){
                line = reader.readLine();
            }
            return reader.readLine();
        }catch (Exception e){
            System.err.println("Error reading file: " + e.getMessage());
        }
        return null;
        //return row;


    }
    //appends line to end of csv file
    public static void writeRow(String filename, float[] row) {
        try (FileWriter writer = new FileWriter(filename, true)) { // Append mode
        writer.write(Arrays.toString(row).replaceAll("[\\[\\]]", "")); // Remove brackets
        writer.write("\n");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    //clears file
    public static void clearFile(String filename) {
        try (FileWriter writer = new FileWriter(filename, false)) {
            writer.write("");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //Returns a 2d array index 0 is the pixel values index 1 is the array version of the answer
    public static float[][] toFloatArray(String s) {
        String[] split = s.split(",");
        float[] array = new float[split.length];
        for (int i = 0; i < split.length; i++) {
            array[i] = Float.parseFloat(split[i]);
        }
        float[] answer = new float[10];
        for (int i = 0; i < 10; i++) {
            if (Float.parseFloat(split[0]) == i) {
                answer[i] = 1;
            }
            else{
                answer[i] = 0;
            }
        }
        return new float[][]{array, answer};
    }

    //Saves the weights and biases of each neuron in each layer
    public static void saveInfo(NeuralNetwork nn, int type) {
        // Write weights and biases to the CSV file using writeRow
        String filename = "";
        switch (type) {
            case 1:
                filename = "src/savedData.csv";
                break;
            case 2:
                filename = "src/savedDataIDX.csv";
                break;
            case 3:
                filename = "src/savedDataJPG.csv";
                break;
            default:
                System.err.println("Invalid type");
                return;
        }
        // Get the weights and biases for each layer
        clearFile(filename);
        float[][] weights1 = nn.getHiddenWeights(1);
        float[] biases1 = nn.getHiddenBiases(1);

        float[][] weights2 = nn.getHiddenWeights(2);
        float[] biases2 = nn.getHiddenBiases(2);

        float[][] weights3 = nn.getHiddenWeights(3);
        float[] biases3 = nn.getHiddenBiases(3);
        // Save weights for layer 1
        for (float[] weightRow : weights1) {
            writeRow(filename, weightRow); // Write weights row by row
        }
        writeRow(filename, new float[0]); // Add an empty line between layers

        // Save weights for layer 2
        for (float[] weightRow : weights2) {
            writeRow(filename, weightRow); // Write weights row by row
        }
        writeRow(filename, new float[0]); // Add an empty line between layers

        // Save weights for layer 3
        for (float[] weightRow : weights3) {
            writeRow(filename, weightRow); // Write weights row by row
        }
        writeRow(filename, new float[0]); // Add an empty line between layers

        // Save biases for layer 1
        writeRow(filename, biases1); // Write biases for layer 1
        writeRow(filename, new float[0]); // Add an empty line between layers

        // Save biases for layer 2
        writeRow(filename, biases2); // Write biases for layer 2
        writeRow(filename, new float[0]); // Add an empty line between layers

        // Save biases for layer 3
        writeRow(filename, biases3); // Write biases for layer 3
    }

    public static Neuron[][] LoadNetworkLT(String filename) {
        // Create a list to store all lines from the file
        //clearFile(filename);
        List<String> lines = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            System.err.println("Error reading file(File manager): " + e.getMessage());
            return null;
        }

        // Process the data to extract weights and biases for each layer
        List<List<float[]>> allWeights = new ArrayList<>();
        List<float[]> allBiases = new ArrayList<>();
        List<float[]> currentLayerWeights = new ArrayList<>();

        int lineIndex = 0;

        // Extract weights for all three layers
        for (int layer = 0; layer < 3; layer++) {
            currentLayerWeights = new ArrayList<>();

            // Read until empty line
            while (lineIndex < lines.size() && !lines.get(lineIndex).isEmpty()) {
                String line = lines.get(lineIndex);
                float[] weights = parseFloatArray(line);
                currentLayerWeights.add(weights);
                lineIndex++;
            }

            // Skip empty line
            lineIndex++;

            allWeights.add(currentLayerWeights);
        }

        // Extract biases for all three layers
        for (int layer = 0; layer < 3; layer++) {
            // Read biases
            if (lineIndex < lines.size() && !lines.get(lineIndex).isEmpty()) {
                String line = lines.get(lineIndex);
                float[] biases = parseFloatArray(line);
                allBiases.add(biases);
                lineIndex++;
            }

            // Skip empty line
            lineIndex++;
        }

        // Create neurons for each layer
        Neuron[][] network = new Neuron[3][];

        for (int layer = 0; layer < 3; layer++) {
            List<float[]> layerWeights = allWeights.get(layer);
            float[] layerBiases = allBiases.get(layer);

            network[layer] = new Neuron[layerWeights.size()];

            for (int neuronIndex = 0; neuronIndex < layerWeights.size(); neuronIndex++) {
                float[] weights = layerWeights.get(neuronIndex);
                float bias = layerBiases[neuronIndex];

                network[layer][neuronIndex] = new Neuron(weights, bias);
            }
        }

        return network;
    }

    private static float[] parseFloatArray(String line) {
        if (line.isEmpty()) {
            return new float[0];
        }

        String[] values = line.split(",");
        float[] result = new float[values.length];

        for (int i = 0; i < values.length; i++) {
            try {
                result[i] = Float.parseFloat(values[i]);
            } catch (NumberFormatException e) {
                System.err.println("Error parsing float: " + values[i]);
                result[i] = 0.0f; // Default value in case of error
            }
        }
        return result;
    }

    public static float[] jpgToFloat(String imagePath) throws IOException{
        BufferedImage image = ImageIO.read(new File(imagePath));
        int width = image.getWidth();
        int height = image.getHeight();
        float[] floatArray = new float[width*height];
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);

                // Extract RGB values
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;

                // Convert to grayscale using the weighted average formula
                float grayscale = 0.299f * red + 0.587f * green + 0.114f * blue;

                // Store the grayscale value (0-255 range) directly in the float array
                floatArray[index++] = grayscale;
            }
        }

        return floatArray;
    }


}
