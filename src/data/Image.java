package data;

/**
 * Represents a single labeled image used in a convolutional neural network (CNN).
 *
 * This class stores the pixel data for an image as a 2D array of doubles, along with
 * the integer label that identifies what the image represents (for example, a digit
 * in a handwritten-digit dataset). Instances of this class serve as data containers
 * that pair input data with the correct target label during training or evaluation.
 */
public class Image {

    private double[][] data; // 2D array/matrix containing data
    private int label; // label for each image/number

    public double[][] getData() { return data; }
    public int getLabel() { return label; }

    public void setData(double[][] data) { this.data = data; }
    public void setLabel(int label) { this.label = label; }

    /**
     * Creates a new Image instance containing pixel data and its associated label.
     *
     * @param data  a 2D array of doubles representing the pixel grid for the image
     * @param label the numeric label associated with this image (e.g., the digit it represents)
     */
    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    /**
     * Returns a string representation of this Image object.
     * The output includes the label of the image followed by all the pixel values
     * in the 2D data array, row by row. Each pixel value is separated by a comma,
     * and each row is separated by a newline. This method is primarily used for
     * debugging or inspecting the image's contents.
     *
     * Example output for a 2x2 image with label 5:
     * 5,
     * 0.0, 1.0,
     * 0.5, 0.8,
     *
     * @return a string containing the label and the pixel data of the image
     */
    @Override public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(label).append(",\n");

        // Loop through each row of the pixel data
        for (int i = 0; i < data.length; i++) {
            // Loop through each column in the current row
            for (int j = 0; j < data[0].length; j++) {
                // Append each pixel value followed by a comma
                sb.append(data[i][j]).append(", ");
            }
            // After finishing a row, append a newline
            sb.append("\n");
        }
        // Convert StringBuilder to String and return
        return sb.toString();
    }
}
