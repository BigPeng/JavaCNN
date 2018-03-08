# JavaCNN

A Java implementation of Convolutional Neural Network.
This is a mavenized fork of https://github.com/BigPeng/JavaCNN refactored for the intention to use it in productive environments.

Original ideas are take from the DeepLearnToolbox (https://github.com/rasmusbergpalm/DeepLearnToolbox).


## Include in your project

If you use maven, it's simple: Just add the dependency 

    <dependencies>

	    <dependency>
	        <groupId>javacnn</groupId>
	        <artifactId>javacnn</artifactId>
	        <version>0.4</version>
	    </dependency>
	   
    </dependencies>

and ratopi's repository 

	<repositories>
        <repository>
            <id>ratopi.de releases</id>
            <url>http://ratopi.github.io/maven/releases/</url>
            <snapshots>
                <enabled>false</enabled>
            </snapshots>
        </repository>
    </repositories>

to your project's pom.xml.


## Build a CNN

	LayerBuilder builder = new LayerBuilder();
	builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
	builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
	builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
	builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
	builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
	builder.addLayer(Layer.buildOutputLayer(10));
	CNN cnn = new CNN(builder, 50);


## Run on MNIST dataset

For running on MNIST dataset see project https://github.com/ratopi/javacnn.mnist.


## Source Code

Get the source code from github:

	git clone https://github.com/ratopi/JavaCNN.git 


## License

MIT
