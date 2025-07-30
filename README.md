# Open3D: A Modern Library for 3D Data Processing


Open3D is an open-source library that supports rapid development of software
that deals with 3D data. The Open3D frontend exposes a set of carefully selected
data structures and algorithms in both C++ and Python. The backend is highly
optimized and is set up for parallelization. 

This Respository is a extention to the Open3d Libary that added some modified
Algorithms to the Open3d tensor module. 
The main goal is to make the odometry and reconstruction semanticly aware to 
enhance the performence in dynamic enviorments.

This is a private Project only.

## Python

For the Python bindings the projects needs to be compiled from source.
For this follow the build form source intructions of the original Open3d respository. 

For installing the package directly into the enviorment specify the cmake flag
-DPYTHON_EXECUTABLE=path/to/pythonexecutable.
A sample install script is provided that enables some flag recomended for my usecase.

The cuda module is only available under linux at the moment.

# C++
For C++ usage refer to the offical Open3d site



## Citation

Please cite [our work](https://arxiv.org/abs/1801.09847) if you use Open3D.

```bib
@article{Zhou2018,
    author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
    title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
    journal   = {arXiv:1801.09847},
    year      = {2018},
}
```
