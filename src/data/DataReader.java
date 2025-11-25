package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads image data from a CSV file and converts it into Image objects.
 *
 * This class is designed to read the MNIST dataset (or similar CSV-based datasets),
 * where each row represents an image:
 * - The first column contains the label (the digit the image represents).
 * - The remaining columns contain pixel values (flattened 28x28 grid).
 *
 * The readData method will parse each row, convert it to a 28x28 2D double array,
 * and store the corresponding label in an Image object.
 */
public class DataReader {

    // MNIST images are 28x28 pixels
    private final int rows = 28;
    private final int columns = 28;

    /**
     * Reads a CSV file from the given path and converts each row into an Image object.
     *
     * @param path the path to the CSV file containing image data
     * @return a list of Image objects containing the pixel data and labels
     */
    public List<Image> readData(String path) {
        List<Image> images = new ArrayList<>();

        try(BufferedReader dataReader = new BufferedReader(new FileReader(path))) {

            dataReader.readLine(); // skip header (for now)

            String line;

            // Read each line of the CSV file
            while ((line = dataReader.readLine()) != null) {
                // Split the line by commas into an array of strings
                String[] lineItems = line.split(",");

                // Create a 2D array to hold the image pixels
                double[][] data = new double[rows][columns];

                // First column is the label
                int label = Integer.parseInt(lineItems[0]);

                // Counter for iterating over flattened pixel values
                int  i = 1;

                // Fill the 2D pixel array
                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < columns; column++) {
                        data[row][column] = (double) Integer.parseInt(lineItems[i]);
                        i++;
                    }
                }

                // Create a new image object with pixels and label
                images.add(new Image(data, label));
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        // returns a list of image objects
        return images;
    }
}
