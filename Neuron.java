import java.util.Arrays;

public class Neuron {
    float[] weights;
    float bias;
    float currentRawWeightedSum;
    float currentActivatedWeightedSum;
    float[] input;
    public float[] normalizedInputs;
    public float activationZj;
    public Neuron(){
        bias = 0;
        currentRawWeightedSum = 0.0f;
        currentActivatedWeightedSum = 0.0f;
    }

    public Neuron(int numWeights){
        weights = new float[numWeights];
        bias = 0;
        currentRawWeightedSum = 0.0f;
        currentActivatedWeightedSum = 0.0f;
    }

    public Neuron(float[] w, float b){
        weights = w;
        bias = b;
    }

    public float[] getWeights(){
        return weights;
    }
    public float getBias(){
        return bias;
    }
    public float getRawWeightedSum(){
        return currentRawWeightedSum;
    }
    public float getActivatedWeightedSum(){
        return currentActivatedWeightedSum;
    }
    public float getPreciseWeights(int i){
        return weights[i];
    }
    public float getPerciseWeight(int i){
        return weights[i];
    }
    //Set up weights to be added to
    public void initializeWeights(int length){
        weights = new float[length];
    }
    public void setWeights(float[] w){
        weights = w;
    }
    public void setBias(float b){
        bias = b;
    }
    public void setPreciseWeights(int index, float w){
        weights[index] = w;
    }
    public void setWeightedSum(float[] input){
        this.input = input;
        float sum = 0;
        for (int i = 0; i < input.length; i++) {
            sum += weights[i] * input[i];
        }
        sum += bias;
        currentRawWeightedSum = sum;
    }
    public void activateWeightedSum(boolean isLayer1){
        for (int i = 0; i < weights.length; i++) {
            currentActivatedWeightedSum = (float) Math.max(0.01*currentRawWeightedSum, currentRawWeightedSum);
        }
        activationZj = (currentActivatedWeightedSum > 0) ? 1.0f : 0.01f;
    }

    //Takes in inputs, stores them and gets the normalized inputs
    public void setInput(float[] input, boolean isLayer1){
        this.input = input;
        if (isLayer1) {
            normalizedInputs = new float[input.length];
            for (int i = 0; i < input.length; i++) {
                normalizedInputs[i] = input[i] / 255;
            }
        }
    }

    //Takes in input and returns activated weighted sum
    public float getOutput(float[] inputValues, boolean isLayer1){
        if (!isLayer1) {
            setInput(inputValues, false);
            setWeightedSum(input);
            activateWeightedSum(false);
            return currentActivatedWeightedSum;
        }
        else{
            setInput(inputValues, true);
            setWeightedSum(normalizedInputs);
            return currentRawWeightedSum;
        }

    }

    public String toString(){
        return Arrays.toString(weights) + " " + bias;
    }

    //Leaky ReLU derivative
    public float getActivationZj(){
        return activationZj;
    }
}
