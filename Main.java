import java.awt.*;
import java.io.IOException;
import javax.swing.*;
public class Main {
    private static final int hiddenLayerNeurons = 64;
    private static final int enochs = 1000;
    private static final float learningRate = 0.01f;

    public static void main(String[] args){
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new GridLayout(7, 1, 5, 5));
        frame.setSize(800, 400);
        frame.setTitle("Rebira's AI Manager");

        JPanel buffer1 = new JPanel(new GridLayout(1, 3));
        JLabel output = new JLabel("");
        JTextField bufferLabel = new JTextField();
        bufferLabel.setEditable(false);
        JTextField bufferLabel2 = new JTextField();
        bufferLabel2.setEditable(false);
        buffer1.add(bufferLabel);
        buffer1.add(output);
        buffer1.add(bufferLabel2);
        frame.add(buffer1);

        //TRAIN CSV//
        JPanel row1 = new JPanel(new GridLayout(1, 6));
        JButton button1 = new JButton("Train CSV");
        JLabel label1 = new JLabel("Enter amount of CSV images to train on: ");
        JTextField textField1 = new JTextField();
        button1.addActionListener(e -> {
            output.setText("Training...");
            trainCSV(Integer.parseInt(textField1.getText()));
            output.setText("Training Complete!");
        });
        row1.add(label1);
        row1.add(textField1);
        row1.add(button1);
        frame.add(row1);

        //TRAIN IDX//
        JPanel row2 = new JPanel(new GridLayout(1, 6));
        JButton button2 = new JButton("Train IDX");
        JLabel label2 = new JLabel("Enter amount of IDX images to train on: ");
        JTextField textField2 = new JTextField(5);
        button2.addActionListener(e -> {
            output.setText("Training...");
            trainIDX(Integer.parseInt(textField2.getText()));
            output.setText("Training Complete!");
        });
        row2.add(label2);
        row2.add(textField2);
        row2.add(button2);
        frame.add(row2);

        //RUN CSV//
        JPanel row3 = new JPanel(new GridLayout(1, 6));
        JButton button3 = new JButton("Run CSV");
        JLabel label3 = new JLabel("Select the csv file line to run: ");
        JTextField textField3 = new JTextField();
        button3.addActionListener(e -> {
            try {
                int textInt = Integer.parseInt(textField3.getText());
                output.setText(String.valueOf(runCSVFile(textInt)));
            }catch (Exception e1){
                System.out.println("Error reading file(jpanel): " + e1.getMessage());
            }
        });
        row3.add(label3);
        row3.add(textField3);
        row3.add(button3);
        frame.add(row3);

        //RUN IDX//
        JPanel row4 = new JPanel(new GridLayout(1, 6));
        JButton button4 = new JButton("Run IDX");
        JLabel label4 = new JLabel("Enter IDX image to predict: ");
        JTextField textField4 = new JTextField(5);
        button4.addActionListener(e -> {
            try {
                output.setText("Predicted output " + String.valueOf(runIDXFile(Integer.parseInt(textField4.getText()))));
            }catch (Exception e1){
                System.out.println("Error reading file: " + e1.getMessage());
            }
        });
        row4.add(label4);
        row4.add(textField4);
        row4.add(button4);
        frame.add(row4);

        //RUN JPG//
        JPanel row5 = new JPanel(new GridLayout(1, 6));
        JButton button5 = new JButton("Run JPG");
        JLabel label5 = new JLabel("Enter file path for JPG image to predict:");
        JTextField textField5 = new JTextField();
        button5.addActionListener(e -> {
            try {
                output.setText("Predicted output: " + String.valueOf(runJPGFile("testSet/img_" + textField5.getText() + ".jpg")));
            }catch (Exception e1){
                System.out.println("Error reading file: " + e1.getMessage());
            }
        });
        row5.add(label5);
        row5.add(textField5);
        row5.add(button5);
        frame.add(row5);

        JPanel buffer2 = new JPanel(new GridLayout(1, 6));
        frame.add(buffer2);



        frame.setVisible(true);
    }


    public static int runCSVFile(int image) throws IOException {
        try {
            String imagePath = FileManager.readRow("src/mnist_test.csv", image);
            float[] fileInput = FileManager.toFloatArray(imagePath)[0];
            Neuron[][] oldData = FileManager.LoadNetworkLT("src/savedData.csv");
            NeuralNetwork nn = new NeuralNetwork(fileInput, oldData[0], oldData[1], oldData[2]);
            return nn.run();
        }
        catch (Exception e){
            System.out.println("Error reading file(run csv): " + e.getMessage());
        }
        return 0;
    }


    public static int runIDXFile(int image){
        if (image>60000){
            image = 60000;
        }
        if (image<0){
            image = 0;
        }
        float[][][] images;
        float[] fileInput;
        try {
            images = IDX3Reader.readIDX3("train-images.idx3-ubyte");
            fileInput = IDX3Reader.toArray(images[image]);
            Neuron[][] oldData = FileManager.LoadNetworkLT("src/savedData.csv");
            NeuralNetwork nn = new NeuralNetwork(fileInput, oldData[0], oldData[1], oldData[2]);
            return nn.run();
        }catch (IOException e){
            System.out.println("Error reading file: " + e.getMessage());
        }
        return  0;
    }


    public static int runJPGFile(String image) throws IOException {
        float[] fileInput = FileManager.jpgToFloat(image);
        Neuron[][] oldData = FileManager.LoadNetworkLT("src/savedData.csv");
        NeuralNetwork nn = new NeuralNetwork(fileInput, oldData[0], oldData[1], oldData[2]);
        return nn.run();
    }


    public static void trainIDX(int num){
        float[][][] images;
        float[] answers;
        if (num>60000){
            num = 60000;
        }
        if (num<0){
            num = 0;
        }
        for(int i = 0; i<num; i++) {
            try {
                images = IDX3Reader.readIDX3("train-images.idx3-ubyte");
                answers = IDX3Reader.readIDX1("train-labels.idx1-ubyte");
                float[] fileInput = IDX3Reader.toArray(images[i]);
                float curanswer = answers[i]*9;
                float[] answer = new float[10];
                for(int j = 0; j<10; j++){
                    if(j == curanswer){
                        answer[j] = 1;
                    }
                    else{
                        answer[j] = 0;
                    }
                }
                if (i == 0 && FileManager.readRow("src/savedDataIDX.csv", 0) == null) {
                    NeuralNetwork nn = new NeuralNetwork(fileInput, answer, hiddenLayerNeurons, 10, enochs, learningRate);
                    nn.train();
                    FileManager.saveInfo(nn, 2);
                } else {
                    Neuron[][] oldData = FileManager.LoadNetworkLT("src/savedDataIDX.csv");
                    NeuralNetwork nn = new NeuralNetwork(fileInput, oldData[0], oldData[1], oldData[2], answer, hiddenLayerNeurons, 10, enochs, learningRate);
                    nn.train();
                    FileManager.saveInfo(nn, 2);
                }
            } catch (Exception e) {
                System.out.println("Error reading file: " + e.getMessage());
            }
        }
    }


    public static void trainCSV(int num){
        for (int i = 0; i<num; i++) {
            String fileText = FileManager.readRow("src/mnist_train.csv", i);
            float[] fileInput = FileManager.toFloatArray(fileText)[0];
            float[] answer = FileManager.toFloatArray(fileText)[1];
            if (i == 1 && FileManager.readRow("src/savedData.csv", 0) == null) {
                NeuralNetwork nn = new NeuralNetwork(fileInput, answer, hiddenLayerNeurons, 10, enochs, learningRate);
                nn.train();
                FileManager.saveInfo(nn, 1);
            } else {
                Neuron[][] oldData = FileManager.LoadNetworkLT("src/savedData.csv");
                NeuralNetwork nn = new NeuralNetwork(fileInput, oldData[0], oldData[1], oldData[2], answer, hiddenLayerNeurons, 10, enochs, learningRate);
                nn.train();
                FileManager.saveInfo(nn, 1);
            }
            System.out.println(i + "/" + num);
        }
        System.out.println("Complete!");
    }
}
