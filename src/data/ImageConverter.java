package data;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageConverter {
    private final int size;

    public ImageConverter(int size){
        this.size = size;
    }

    public Image getImage(String file, int label) throws IOException {
        File input = new File(file);

        BufferedImage originalImage = ImageIO.read(input);

        BufferedImage resizedImage = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = resizedImage.createGraphics();
        graphics2D.drawImage(originalImage, 0, 0, size, size, null);
        graphics2D.dispose();

        // Convert to grayscale and place it in the data buffer
        double[][] data = new double[size][size];
        for (int y = 0; y < resizedImage.getHeight(); y++) {
            for (int x = 0; x < resizedImage.getWidth(); x++) {
                int p = resizedImage.getRGB(x, y);

                int a = (p >> 24) & 0xff;
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;

                // Calculate average
                int avg = (r + g + b) / 3;

                // Replace RGB value with avg
                p = (a << 24) | (avg << 16) | (avg << 8) | avg;

                resizedImage.setRGB(x, y, p); // generate the greyscale image.
                data[x][y] = avg;
            }
        }
        return new Image(data, label);
    }
}
