#include <iostream>
#include <curl/curl.h>

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main() {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        std::cout << readBuffer << std::endl;
    }
    return 0;
}

/*
PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\NLP\NLP_API_VyoriusTest\test_libcurl_nlohmannJson> g++ .\test_libcurl_with_ollama.cpp -o .\test_libcurl_with_ollama.exe -I"C:\Users\hi\dev\vcpkg\installed\x64-windows\include" -L"C:\Users\hi\dev\vcpkg\installed\x64-windows\lib" -lcurl
PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\NLP\NLP_API_VyoriusTest\test_libcurl_nlohmannJson> .\test_libcurl_with_ollama.exe
Ollama is running
PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\NLP\NLP_API_VyoriusTest\test_libcurl_nlohmannJson>

*/