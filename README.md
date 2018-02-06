# Installation of OpenCV 3.4.0 on Windows, compatible with VC++ 2015 or 2017

Step 1: Get the chocolatey package manager. Find the instructions at https://chocolatey.org/

Step 2: Open a command prompt as an administrator. Install openCV with `choco install -y opencv`. At the time of writing, chocolatey has version `3.4.0`

## Manual Install
Alternatively, install OpenCV from `https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.0/opencv-3.4.0-vc14_vc15.exe/download`

## OpenCV with extra goodies and Debug Libs
The chocolatey version *doesn't* come with all the extra modules and non-free algorithms. If you need those, or want the debug version of OpenCV, download this package: `https://drive.google.com/file/d/193m8zam2k2ybu3BZ2iF2me07hrGS5_Iw/view`.

This package only supports VC15/Visual Studio 2017. Drop the `tools` folder from the 7z file in to C:, and follow the instructions from step 3 for VS2017.

Step 3: Install Visual Studio  2015 or 2017. Ensure you have "Desktop Development with C++" support enabled.

Step 4: Open a command prompt as an administrator and run the following command to set your environment variables. 

This is assuming OpenCV is installed to `C:\tools`. If you installed OpenCV to a different directory, replace `C:\tools` in the commands below.

		VS 2015: setx -m OPENCV_DIR C:\tools\opencv\build\x64\vc14
		VS 2017: setx -m OPENCV_DIR C:\tools\opencv\build\x64\vc15

With the previous command window, set your PATH with the following command:

		setx -m PATH "%PATH%;%OPENCV_DIR%\bin"

Step 5: Probably a good idea to restart your computer at this point

##  Visual Studio Project Creation

If you have Visual Studio 2017 or 2015, the project file in this repository is mostly set up already. Make a copy of it. You can build for "Debug" or "Release". 64-bit mode is the only one supported

Step 1: Open the solution file.

Step 2: Right click on "Project 1" in the solution explorer.

Step 3: Under `Configuration Properties -> General` open the `Windows SDK` dropdown menu.

Step 4: Select the first item on the list.

Step 5: Press "OK"

You should now be able to compile your programs.

### Working Directory

By default, visual studio uses the directory containing the `.vcxproj` file as the working directory. 

In this case, that directory is called "Project1"

You'll need to place any resource files in that directory to be able to access them using relative paths in your program

# Alternate IDEs (You don't need Visual Studio)

If you want to use these libraries, but don't want to install visual studio, you can install the microsoft compiler and runtime by itself:

[Visual C++ 2017 Build Tools](https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15)
[Visual C++ 2017 Runtime](https://aka.ms/vs/15/release/VC_redist.x64.exe)

Instructions for Eclipse and the Microsoft Compiler: http://codewriterstips.blogspot.ca/2012/05/using-microsoft-c-compiler-with-eclipse.html

I will post exact instructions for eclipse soon. For now, here are the required paths for eclipse. This will require %PATH% and %OPENCV_DIR% to be set correctly.

Additional Include Directories: $(OPENCV_DIR)\..\..\include
Library Search Path: $(OPENCV_DIR)\lib
Additional Dependencies (Debug): opencv_world340d
Additional Dependencies (Release): opencv_world

## Sources
https://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html
https://docs.opencv.org/2.4/doc/tutorials/introduction/windows_visual_studio_Opencv/windows_visual_studio_Opencv.html#windows-visual-studio-how-to
