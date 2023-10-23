package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {

    private static final int columns = 28;
    private static final int rows = 28;

    public static List<Image> readData(String filePath) {
        List<Image> images = new ArrayList<>();

        try(BufferedReader dataReader = new BufferedReader(new FileReader(filePath))){
            String line;

            while((line = dataReader.readLine())!=null){
                String[] imagePart = line.split(",");

                double[][] data = new double[rows][columns];
                int label = Integer.parseInt(imagePart[0]);

                int i=1;

                for(int row = 0; row< rows; row++)
                    for(int col = 0; col< columns; col++){
                        data[row][col] =  Integer.parseInt(imagePart[i]);
                        i++;
                    }

                images.add(new Image(data, label));
            }

        }catch(Exception e){
            System.out.println(e.getMessage());
        }

        return images;
    }
}
