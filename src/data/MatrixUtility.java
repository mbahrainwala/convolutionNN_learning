package data;

public class MatrixUtility {

    private MatrixUtility(){};
    public static double[][] add(double[][] a, double[][] b){
        double[][] out = new double[a.length][a[0].length];

        for(int r=0; r<a.length; r++){
            for(int c=0; c<a[0].length; c++){
                out[r][c]=a[r][c]+b[r][c];
            }
        }

        return out;
    }

    public static double[] add(double[] a, double[] b){
        double[] out = new double[a.length];

        for(int c=0; c<a.length; c++){
            out[c]=a[c]+b[c];
        }

        return out;
    }

    public static double[][] multiply(double[][] a, double scalar){
        double[][] out = new double[a.length][a[0].length];

        for(int r=0; r<a.length; r++){
            for(int c=0; c<a[0].length; c++){
                out[r][c]=a[r][c] * scalar;
            }
        }

        return out;
    }

    public static double[] multiply(double[] a, double scalar){
        double[] out = new double[a.length];

        for(int c=0; c<a.length; c++){
            out[c]=a[c]*scalar;
        }

        return out;
    }

    public static double[][] flipMatrixHorizontal(double[][] input){
        int rows = input.length;
        int cols = input[0].length;

        double[][] out = new double[rows][cols];

        for(int i=0; i< rows; i++)
            for(int j=0; j< cols; j++)
                out[rows-i-1][j]=input[i][j];

        return out;
    }

    public static double[][] flipMatrixVertical(double[][] input){
        int rows = input.length;
        int cols = input[0].length;

        double[][] out = new double[rows][cols];

        for(int i=0; i< rows; i++)
            for(int j=0; j< cols; j++)
                out[i][cols-j-1]=input[i][j];

        return out;
    }
}
