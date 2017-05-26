Data mining project
===================

This is the actual source code of the Data Mining project.

Project structure
-----------------

This project contains some utility classes, under package
`it.unipd.dei.dm1617`, while the actual implemented code is under
`it.unipd.dei.dm1617.release`.
The build configuration is defined in the file `build.gradle`, along
with all the dependencies of the project.

Linux/MacOS build
------------------------------

To compile the code under Linux\MaxOS, you just have to run the
following command

    ./gradlew compileJava
    
To run the main class, you need to set up the Java classpath properly,
including all the dependencies of the project. The following command
will print a colon separated list of the paths to the dependencies of
the project

    ./gradlew showDepsClasspath
    
You can store the result in an environment variable as follows

    export CP=$(./gradlew showDepsClasspath | grep jar)
    
Finally, to run the project's main class you can use the following command

    java -Dspark.master=local[4] -cp $CP:build/classes/main it.unipd.dei.dm1617.release.DMProject medium-sample.dat.bz2

The final argument specifies the path to the dataset.


Saving the lemmatization
------------------------------

The two lines of commented code found in the block starting at line 112 
in file "DMProject.java" can be used to, respectively, save and load the
lemmatized dataset, so as you don't have to recompute the lemmatization 
at each run.

Note that line 113 should only be called on the first time you run the main
along with lines 107 and 110: after the lemmatized pages have been saved,
these three lines can be commented out, while line 114 can be uncommented.

