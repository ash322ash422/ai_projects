#include <iostream>
#include <string>
#include <sstream>

#include <curl/curl.h> //external libraries that needs to be installed
#include <nlohmann/json.hpp> //external libraries that needs to be installed

// For convenience
using json = nlohmann::json;
using namespace std;

// Ollama API endpoint and model name
const string OLLAMA_API_URL = "http://localhost:11434/api/generate"; // make sure this is running locally
const string MODEL_NAME = "llama3.2";

// Callback function to handle streaming response from the server
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Function to get response from the Ollama server
string get_response(const string& prompt, const string& model = MODEL_NAME) {
    CURL* curl;
    CURLcode res;
    string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        // Set the URL
        curl_easy_setopt(curl, CURLOPT_URL, OLLAMA_API_URL.c_str());

        // Construct the JSON payload
        json payload = {
            {"model", model},
            {"prompt", prompt},
            {"stream", true}
        };

        // Convert payload to string
        string payload_str = payload.dump();

        // Set the POST data
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());

        // Set headers
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Set the callback function to handle the response
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        // Perform the request
        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            cerr << "Error: " << curl_easy_strerror(res) << endl;
        }

        // Clean up
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    }

    curl_global_cleanup();

    // Parse the response and extract the output
    string output;
    try {
        stringstream ss(readBuffer);
        string line;
        while (getline(ss, line)) {
            if (!line.empty()) {
                json data = json::parse(line);
                if (data.contains("response")) {
                    output += data["response"].get<string>();
                }
            }
        }
    } catch (const json::parse_error& e) {
        return "An error occurred while parsing the response.";
    }

    return output;
}

// Main function
int main() {
    cout << "\nWelcome to the Local LLM Console Application (Powered by Ollama)" << endl;
    cout << "Type 'exit' to quit the application.\n" << endl;

    string user_input;

    while (true) {
        cout << "You: ";
        getline(cin, user_input);

        if (user_input == "exit") {
            cout << "Goodbye!" << endl;
            break;
        }

        string response = get_response(user_input);
        cout << "AI: " << response << "\n" << endl;
    }

    return 0;
}

/* // sample run:
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
*/