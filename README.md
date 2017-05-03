Data mining project
===================

This is a skeleton project you can use to start working on your own
project.

Project structure
-----------------

This project contains some utility classes, under package
`it.unipd.dei.dm1617`, and some examples, under package
`it.unipd.dei.dm1617.examples`.
The build configuration is defined in the file `build.gradle`, along
with all the dependencies of the project.

Data format
-----------

While Spark accepts several data formats, this project is set up so to
work with files text formatted as follows. Each line represents a
single Wikipedia page and is a well-formed JSON object with the
following fields:

 - id: a unique integer id for the page
 - title: the title of the page
 - categories: a list of strings
 - text: the actual text of the page
 
These files can be compressed in a variety of formats, which Spark
handles transparently. In particular, the datasets you can download
are compressed using the bzip2 library. On Windows it's not possible
to uncompress files compressed in this way out of the box, however
Spark can work with them nonetheless. On Linux, you can inspect the
contents of these files using the `bzless` command line utility.

This project stub provides the class `it.unipd.dei.dm1617.InputOutput`
to load and write files with this format. The static methods of this
class yield and require datasets of `it.unipd.dei.dm1617.WikiPage`
objects. Refer the documentation of these classes for further details.

Linux/MacOS build and examples
------------------------------

To compile the code under Linux\MaxOS, you just have to run the
following command

    ./gradlew compileJava
    
To run the examples, you need to set up the Java classpath properly,
including all the dependencies of the project. The following command
will print a colon separated list of the paths to the dependencies of
the project

    ./gradlew showDepsClasspath
    
You can store the result in an environment variable as follows

    export CP=$(./gradlew showDepsClasspath | grep jar)
    
which you can then include in your Java invocation as follows

    java -Dspark.master=local -cp $CP:build/classes/main it.unipd.dei.dm1617.examples.Sample medium-sample.dat.bz2 small-sample.dat.bz2 0.1

note that `build/classes/main` is included in the classpath as well,
since it includes the actual compiled classes of the project.

Windows build and examples.
---------------------------

For building the project, you can use the following command

    .\gradlew.bat compileJava

For running the program, Windows requires a little more
configuration. In order for everything to work properly, you will need
the `winutils.exe` executable from the Hadoop distribution used by
Spark. You can get a pre-built binary from this link
https://github.com/steveloughran/winutils/raw/master/hadoop-2.7.1/bin/winutils.exe
Once you have downloaded this executable, place in a subdirectory
named `bin` of a directory of your choosing. For instance, on my
system I placed the `winutils.exe` executable in the following path

    C:\Users\ceccarel\Desktop\hadoop\bin\winutils.exe
    
Now, whenever you run your Spark application, set the
`hadoop.home.dir` java property to the path containing `bin\winutils.exe`.

As for running, your best option is to configure your favourite Java
IDE appropriately and run your programs from there.
    
Further reading
---------------

 - Spark Machine Learning documentation: http://spark.apache.org/docs/latest/mllib-guide.html#mllib-rdd-based-api
   There are two API flavors for the Machine Learning library of
   Spark: the older one based directly on RDDs, and the newer one
   based on the concept of DataFrames. For simplicity and consistency
   with the lessons we stick with the RDD based API.
 - Various information retrieval topics are available in this book: https://nlp.stanford.edu/IR-book/html/htmledition/contents-1.html
