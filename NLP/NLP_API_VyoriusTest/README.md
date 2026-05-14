# SETUP INSTRUCTIONS

Download visual studio code (community edition is free) from https://visualstudio.microsoft.com/downloads/

  * Execute the downloaded file : VisualStudioSetup.exe and run it.
  * During installation, select the "Desktop development with C++" workload. This installs all the necessary compilers, libraries, and tools. Takes 10-15 minutes

Install `vcpkg` . This is a package manager for Visual Studio C++

Using vcpkg, install curl and nlohmann-json.

Compile and run this in Visual Studio

# Thought process for this project

1) Run LLAMA3.2 on local machine using ollama on port 11434

2) First design a quick working prototype in python. 

3) Then convert this code into C++. Install the necessary libraries like curl and nlohmann-json. Test this libraries with C++ code located in directory test_libcurl_nlohmannJson

4) Design the talk_to_llama3_with_API.cpp. Compile and run this.

5) Optimize the code.

Sample run:

```
PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\NLP\NLP_API_VyoriusTest> g++ .\talk_to_llama3_via_API.cpp -o talk_to_llama3_via_API.exe -I"C:\Users\hi\dev\vcpkg\installed\x64-windows\include" -L"C:\Users\hi\dev\vcpkg\installed\x64-windows\lib" -lcurl
PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\NLP\NLP_API_VyoriusTest> .\talk_to_llama3_via_API.exe

Welcome to the Local LLM Console Application (Powered by Ollama)
Type 'exit' to quit the application.

You: Tell me a joke
AI: Here's one:

What do you call a fake noodle?

An impasta.

You: Finish following sentence: I woke up to a sunny morning, so I went out for a
AI: ...run along the nearby beach, feeling invigorated and ready to tackle the day.

You: exit
Goodbye!
PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\NLP\NLP_API_VyoriusTest>
```

