*) Download visual studio code (community edition is free) from https://visualstudio.microsoft.com/downloads/
  - Execute the downloaded file : VisualStudioSetup.exe and run it.
  - During installation, select the "Desktop development with C++" workload. This installs all the necessary compilers, libraries, and tools. Takes 10-15 minutes

1) Now we download vcpkg. This is a package manager which we use to download other packages like libcurl,nlohmann-json,etc. Create a directory `C:\Users\hi\dev>` and run commend:
`
PS C:\Users\hi\dev> git clone https://github.com/microsoft/vcpkg.git
Cloning into 'vcpkg'...
remote: Enumerating objects: 256273, done.
remote: Counting objects: 100% (11725/11725), done.
remote: Compressing objects: 100% (645/645), done.
remote: Total 256273 (delta 11345), reused 11234 (delta 11080), pack-reused 244548 (from 1)
Receiving objects: 100% (256273/256273), 79.21 MiB | 6.48 MiB/s, done.
Resolving deltas: 100% (169122/169122), done.
Updating files: 100% (11899/11899), done.
`

2)` PS C:\Users\hi\dev> cd vcpkg '

3.1)
`
PS C:\Users\hi\dev\vcpkg> .\bootstrap-vcpkg.bat
Downloading https://github.com/microsoft/vcpkg-tool/releases/download/2024-11-12/vcpkg.exe -> C:\Users\hi\dev\vcpkg\vcpkg.exe... done.
Validating signature... done.

vcpkg package management program version 2024-11-12-eb492805e92a2c14a230f5c3deb3e89f6771c321

See LICENSE.txt for license information.
Telemetry
---------
vcpkg collects usage data in order to help us improve your experience.
The data collected by Microsoft is anonymous.
You can opt-out of telemetry by re-running the bootstrap-vcpkg script with -disableMetrics,
passing --disable-metrics to vcpkg on the command line,
or by setting the VCPKG_DISABLE_METRICS environment variable.

Read more about vcpkg telemetry at docs/about/privacy.md
`

3.2)
Now we integrate it system-wide to avoid specifying paths manually when we compile in MSBuild:
`
PS C:\Users\hi\dev\vcpkg> .\vcpkg.exe integrate install
Applied user-wide integration for this vcpkg root.
CMake projects should use: "-DCMAKE_TOOLCHAIN_FILE=C:/Users/hi/dev/vcpkg/scripts/buildsystems/vcpkg.cmake"

All MSBuild C++ projects can now #include any installed libraries. Linking will be handled automatically. Installing new libraries will make them instantly available.
`

4) Now we install libcurl: 
NOTE: if you do not have Visual Studio installed, then this step would give you following error when you execute ".\vcpkg.exe install curl[tool]":
`
error: in triplet x64-windows: Unable to find a valid Visual Studio instance
Could not locate a complete Visual Studio instance
`
`
PS C:\Users\hi\dev\vcpkg> .\vcpkg.exe install curl[tool]
Computing installation plan...
A suitable version of cmake was not found (required v3.30.1).
Downloading cmake-3.30.1-windows-i386.zip
Successfully downloaded cmake-3.30.1-windows-i386.zip.
Extracting cmake...
A suitable version of 7zip was not found (required v24.9.0).
Downloading 7z2409-extra.7z
Successfully downloaded 7z2409-extra.7z.
Extracting 7zip...
A suitable version of 7zr was not found (required v24.9.0).
Downloading 44D8504A-7zr.exe
Successfully downloaded 44D8504A-7zr.exe.
The following packages will be built and installed:
    curl[core,non-http,schannel,ssl,sspi,tool]:x64-windows@8.11.0#1
   vcpkg-cmake:x64-windows@2024-04-23
   vcpkg-cmake-config:x64-windows@2024-05-23
   zlib:x64-windows@1.3.1
Additional packages (*) will be modified to complete this operation.
Detecting compiler hash for triplet x64-windows...
A suitable version of powershell-core was not found (required v7.2.24).
Downloading PowerShell-7.2.24-win-x64.zip
Successfully downloaded PowerShell-7.2.24-win-x64.zip.
Extracting powershell-core...
Compiler found: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe
Restored 4 package(s) from C:\Users\hi\AppData\Local\vcpkg\archives in 473 ms. Use --debug to see more details.
Installing 1/4 vcpkg-cmake:x64-windows@2024-04-23...
Elapsed time to handle vcpkg-cmake:x64-windows: 17.7 ms
vcpkg-cmake:x64-windows package ABI: 132bb3bb19019c90251f67d44707dcdf7be82031283edf79bb76930f6914b884
Installing 2/4 vcpkg-cmake-config:x64-windows@2024-05-23...
Elapsed time to handle vcpkg-cmake-config:x64-windows: 20.4 ms
vcpkg-cmake-config:x64-windows package ABI: 5bd5d5697893516dfbac367e9e009dbc51b36535c34d6e5b45c23e36cdd3c399
Installing 3/4 zlib:x64-windows@1.3.1...
Elapsed time to handle zlib:x64-windows: 26 ms
zlib:x64-windows package ABI: cd775d1b5ac16149c742939e360d51422266c272bce6735678c8c9afe486ea42
Installing 4/4 curl[core,non-http,schannel,ssl,sspi,tool]:x64-windows@8.11.0#1...
Elapsed time to handle curl:x64-windows: 69.8 ms
curl:x64-windows package ABI: ece9e017fed322221c43075596c4a2270295da67097eda5349604fe9d64b2ffc
Total install time: 138 ms
curl is compatible with built-in CMake targets:

    find_package(CURL REQUIRED)
    target_link_libraries(main PRIVATE CURL::libcurl)

`

(This step could take 2-5 minutes)

5) Now install nlohmann-json:
`
PS C:\Users\hi\dev\vcpkg> .\vcpkg.exe install nlohmann-json
Computing installation plan...
The following packages will be built and installed:
    nlohmann-json:x64-windows@3.11.3#1
Detecting compiler hash for triplet x64-windows...
Compiler found: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe
Restored 1 package(s) from C:\Users\hi\AppData\Local\vcpkg\archives in 406 ms. Use --debug to see more details.
Installing 1/1 nlohmann-json:x64-windows@3.11.3#1...
Elapsed time to handle nlohmann-json:x64-windows: 39.2 ms
nlohmann-json:x64-windows package ABI: c5d115e736fe6ade477c0b17b63529f8ea7f58b3a0109a6bdd297a60276e8fd9
Total install time: 39.5 ms
The package nlohmann-json provides CMake targets:

    find_package(nlohmann_json CONFIG REQUIRED)
    target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)

The package nlohmann-json can be configured to not provide implicit conversions via a custom triplet file:

    set(nlohmann-json_IMPLICIT_CONVERSIONS OFF)

For more information, see the docs here:

    https://json.nlohmann.me/api/macros/json_use_implicit_conversions/
`