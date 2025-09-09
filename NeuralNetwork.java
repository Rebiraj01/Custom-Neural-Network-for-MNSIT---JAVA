import java.util.*;
public class NeuralNetwork {
    private final float[] correctArray;
    private final int layerSize;
    private final int outputSize;
    private final int epochs;
    private final float learningRate;
    private final int numInputs;
    private final float[] input;
    private Neuron[] layer1;
    private Neuron[] layer2;
    private Neuron[] outputLayer;
    //Used to first run to set up layers and weights (TRAIN)
    public NeuralNetwork(float[] inputs, float[] CA, int WLS, int WOS, int e, float LR){ //Used for training
        input = inputs;
        numInputs = inputs.length;
        correctArray = CA;
        layerSize = WLS;
        outputSize = WOS;
        epochs = e;
        learningRate = LR;
        layer1 = new Neuron[layerSize];
        layer2 = new Neuron[layerSize];
        outputLayer  = new Neuron[outputSize];

        //Initializing Layers
        for (int i = 0; i < layerSize; i++) {
            layer1[i] = new Neuron(numInputs);
        }
        for (int i = 0; i < layerSize; i++){
            layer2[i] = new Neuron(layerSize);
        }
        for (int i = 0; i < outputSize; i++) {
            outputLayer[i] = new Neuron(layerSize);
        }

        heInitialization(); //Setup of original weights and biases
    }

    //Used to continue with previous run info (TRAIN)
    public NeuralNetwork(float[] inputs, Neuron[] layer1, Neuron[] layer2, Neuron[] outputLayer, float[] CA, int WLS, int WOS, int e, float LR){//Used to test
        input = inputs;
        numInputs = inputs.length;
        correctArray = CA;
        layerSize = WLS;
        outputSize = WOS;
        epochs = e;
        learningRate = LR;
        this.layer1 = layer1;
        this.layer2 = layer2;
        this.outputLayer = outputLayer;
    }

    //Used to tun Neural Network (RUN)
    public NeuralNetwork(float[] inputs, Neuron[] layer1, Neuron[] layer2, Neuron[] outputLayer){
        input = inputs;
        numInputs = inputs.length;
        layerSize = layer1.length;
        outputSize = outputLayer.length;
        epochs = 0;
        learningRate = 0;
        this.layer1 = layer1;
        this.layer2 = layer2;
        this.outputLayer  = outputLayer;
        correctArray = new float[outputSize];
    }
    //--------------------------//
    //Manage Weights and Biases//
    //------------------------//

    //Getters
    public float[][] getHiddenWeights(int layer){
        float[][] weights = new float[getLayerSize(layer)][getNumWeights(layer)];
        if (layer==1) {
            for (int i = 0; i < layerSize; i++) {
                for (int j = 0; j < numInputs; j++) {
                    weights[i][j] = layer1[i].getWeights()[j];
                }
            }
        }
        else if (layer==2) {
            for (int i = 0; i < layerSize; i++) {
                for (int j = 0; j < layerSize; j++) {
                    weights[i][j] = layer2[i].getWeights()[j];
                }
            }
        }
        else if (layer==3) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < layerSize; j++) {
                    weights[i][j] = outputLayer[i].getWeights()[j];
                }
            }
        }
        return weights;
    }

    public float[] getHiddenBiases(int layer){
        float[] biases = new float[getLayerSize(layer)];
        if (layer==1) {
            for (int i = 0; i < layerSize; i++) {
                biases[i] = layer1[i].getBias();
            }
        }
        if (layer==2) {
            for (int i = 0; i < layerSize; i++) {
                biases[i] = layer2[i].getBias();
            }
        }
        if (layer==3) {
            for (int i = 0; i < outputSize; i++) {
                biases[i] = outputLayer[i].getBias();
            }
        }
        return biases;
    }

    public int getLayerSize(int layer){
        return switch(layer){
            case 1 ->  layerSize;
            case 2 ->  layerSize;
            case 3 ->  outputSize;
            default ->  0;
        };
    }

    public int getNumWeights(int layer){
        return switch (layer) {
            case 1 -> numInputs;
            case 2 -> layerSize;
            case 3 -> layerSize;
            default -> 0;
        };
    }

    //Setters
    public void setWeights(int layer, float[] weights){
        if (layer==1) {
            for (int i = 0; i < layerSize; i++) {
                layer1[i].setWeights(new float[]{weights[i]});
            }
        }
        if (layer==2) {
            for (int i = 0; i < outputSize; i++) {
                layer2[i].setWeights(new float[]{weights[i]});
            }
        }

    }

    public void setBiases(int layer, float[] biases){
        if (layer==1) {
            for (int i = 0; i < layerSize; i++) {
                layer1[i].setBias(biases[i]);
            }
        }
        if (layer==2) {
            for (int i = 0; i < outputSize; i++) {
                layer2[i].setBias(biases[i]);
            }
        }
    }

    //Initialize weights and biases (biases set to 0)
    public void heInitialization(){
        final float stdDev1 = (float) Math.sqrt(2.0/numInputs);
        final float stdDev2 = (float) Math.sqrt(2.0/layerSize);
        final float stdDev3 = (float) Math.sqrt(2.0/layerSize);
        Random rand = new Random();
        for (int i = 0; i<layerSize; i++){
            layer1[i].setBias(0);
            layer1[i].initializeWeights(numInputs);
            for (int j = 0; j<numInputs; j++){
                layer1[i].setPreciseWeights(j, (float) rand.nextGaussian()*stdDev1);
            }
        }
        for (int i = 0; i<layerSize; i++){
            layer2[i].setBias(0);
            layer2[i].initializeWeights(layerSize);
            for (int j = 0; j<layerSize; j++){
                layer2[i].setPreciseWeights(j, (float) rand.nextGaussian()*stdDev2);
            }
        }
        for (int i = 0; i<outputSize; i++){
            outputLayer[i].setBias(0);
            outputLayer[i].initializeWeights(layerSize);
            for (int j = 0; j<layerSize; j++){
                outputLayer[i].setPreciseWeights(j, (float) rand.nextGaussian()*stdDev3);

            }
        }
    }

    //Actiavte the weighted sum float[] from the output layer
    public static float[] softmax(float[] input) {
        float max = Float.NEGATIVE_INFINITY;

        for(float val : input) {
            if (val > max) {
                max = val;
            }
        }

        float sum = 0.0F;
        float[] output = new float[input.length];

        for(int i = 0; i < input.length; ++i) {
            output[i] = (float)Math.exp((double)(input[i] - max));
            sum += output[i];
        }

        for(int i = 0; i < output.length; ++i) {
            output[i] /= sum;
        }

        return output;
    }


    //-------------//
    //Forward Pass//
    //-----------//

    public float[] forward(){
        //After weights and biases are set up
        float[] layer1Results = new float[layerSize];
        for(int i = 0; i<layerSize; i++){
            layer1Results[i] = layer1[i].getOutput(input, true);
        }

        float[] layer2Results = new float[outputSize];
        for(int i = 0; i<outputSize; i++){
            layer2Results[i] = layer2[i].getOutput(layer1Results, false);

        }
        float[] outputResults = new float[outputSize];
        for(int i = 0; i<outputSize; i++){
            outputResults[i] = outputLayer[i].getOutput(layer2Results, false);
        }

        float[] probabilites =  softmax(outputResults);
        //System.out.println(Arrays.toString(probabilites));
        return probabilites;

    }

    //----------------//
    //backpropagation//
    //--------------//

    public float calculateLoss(float[] output){
        float loss = 0;
        for (int i = 0; i<outputSize; i++){
            loss+= correctArray[i] * (float) Math.log(output[i] + 1e-15f);
        }
        return -1.0f * loss;
    }

    public void backpropagate(float[] output){
        float[] outputErrorTerm = new float[outputSize];
        for (int i = 0; i<outputSize; i++){
            outputErrorTerm[i] = output[i] - correctArray[i];
        }

        float[] layer2ErrorTerm = new float[layerSize];
        for (int i = 0; i<layerSize; i++) {
            float tempSum = 0.0f;
            for (int j = 0; j < outputSize; j++) {
                //Leaky ReLU derivative
                float activationZj = (layer2[i].getActivatedWeightedSum() > 0) ? 1.0f : 0.01f;
                //1 loop of accululated error term
                tempSum += outputErrorTerm[j] * outputLayer[j].getPerciseWeight(i) * activationZj;
            }
            layer2ErrorTerm[i] = tempSum;
        }

        float[] layer1ErrorTerm = new float[layerSize];
        for (int i = 0; i<layerSize; i++) {
            float tempSum = 0.0f;
            for (int j = 0; j < layerSize; j++) {
                //Leaky ReLU derivative
                float activationZj = (layer1[i].getActivatedWeightedSum() > 0) ? 1.0f : 0.01f;
                //1 loop of accululated error term
                tempSum += layer2ErrorTerm[j] * layer2[j].getPerciseWeight(i) * activationZj;
            }
            layer1ErrorTerm[i] = tempSum;
        }

        //Backpropatation to update weights//

        for (int i = 0; i<outputSize; i++){
            for(int j = 0; j<layerSize; j++){
                float weightUpdate = learningRate * outputErrorTerm[i] * layer2[j].getActivatedWeightedSum();
                outputLayer[i].setPreciseWeights(j, outputLayer[i].getPerciseWeight(j) - weightUpdate);
            }
            outputLayer[i].setBias(outputLayer[i].getBias() - learningRate * outputErrorTerm[i]);
        }
        for (int i = 0; i<layerSize; i++){
            for(int j = 0; j<layerSize; j++){
                float weightUpdate = learningRate * layer2ErrorTerm[i] * layer1[j].getActivatedWeightedSum();
                layer2[i].setPreciseWeights(j, layer2[i].getPerciseWeight(j) - weightUpdate);
            }
            layer2[i].setBias(layer2[i].getBias() - learningRate * layer2ErrorTerm[i]);
        }
        for (int i = 0; i<layerSize; i++){
            for(int j = 0; j<numInputs; j++){
                float weightUpdate = learningRate * layer1ErrorTerm[i] * input[j];
                layer1[i].setPreciseWeights(j, layer1[i].getPerciseWeight(j) - weightUpdate);
            }
            layer1[i].setBias(layer1[i].getBias() - learningRate * layer1ErrorTerm[i]);
        }
    }

    public int getMax(float[] output){
        float max = Float.NEGATIVE_INFINITY;
        int maxIndex = 0;
        for (int i = 0; i<outputSize; i++){
            if (output[i] > max){
                max = output[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    //---------//
    //Training//
    //-------//
    public int train(){
        float[] output = new float[outputSize];
        for (int i = 0; i<epochs; i++){
            output = forward();
            backpropagate(output);
        }
        return getMax(output);
    }

    //----//
    //RUN//
    //--//

    public int run(){
        float[] output = forward();
        return getMax(output);
    }


}

