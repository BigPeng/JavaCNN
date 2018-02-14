import edu.hitsz.c102c.cnn.CNN;
import edu.hitsz.c102c.cnn.Layer;
import edu.hitsz.c102c.dataset.Dataset;

/**
 * <p/>
 * Created: 14.02.2018 09:06
 *
 * @author Ralf Th. Pietsch &lt;ratopi@abwesend.de&gt;
 */
public class Test
{
	public static void main( final String[] args )
	{
		final CNN.LayerBuilder builder = new CNN.LayerBuilder();
		builder.addLayer( Layer.buildInputLayer( new Layer.Size( 28, 28 ) ) );
		builder.addLayer( Layer.buildConvLayer( 6, new Layer.Size( 5, 5 ) ) );
		builder.addLayer( Layer.buildSampLayer( new Layer.Size( 2, 2 ) ) );
		builder.addLayer( Layer.buildConvLayer( 12, new Layer.Size( 5, 5 ) ) );
		builder.addLayer( Layer.buildSampLayer( new Layer.Size( 2, 2 ) ) );
		builder.addLayer( Layer.buildOutputLayer( 10 ) );
		final CNN cnn = new CNN( builder, 50 );

		final String fileName = "dataset/train.format";
		final Dataset dataset = Dataset.load( fileName, ",", 784 );
		cnn.train( dataset, 100 );

		final Dataset testset = Dataset.load( "dataset/test.format", ",", -1 );
		cnn.predict( testset, "dataset/test.predict" );
	}
}
