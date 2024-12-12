#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // create a JSON object
    json j =
    {
        {"pi", 3.141},
        {"happy", true},
        {"name", "Niels"},
        {"nothing", nullptr},
        {
            "answer", {
                {"everything", 42}
            }
        },
        {"list", {1, 0, 2}},
        {
            "object", {
                {"currency", "USD"},
                {"value", 42.99}
            }
        }
    };

    // add new values
    j["new"]["key"]["value"] = { "another", "list" };

    // count elements
    auto s = j.size();
    j["size"] = s;

    // pretty print with indent of 4 spaces
    std::cout << std::setw(4) << j << '\n';
}

/*
PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\NLP\NLP_API_VyoriusTest\test_libcurl_nlohmannJson> g++ .\test_nlohmann_json.cpp -o .\test_nlohmann_json.exe -I"C:\Users\hi\dev\vcpkg\installed\x64-windows\include"
PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\NLP\NLP_API_VyoriusTest\test_libcurl_nlohmannJson> .\test_nlohmann_json.exe
{
    "answer": {
        "everything": 42
    },
    "happy": true,
    "list": [
        1,
        0,
        2
    ],
    "name": "Niels",
    "new": {
        "key": {
            "value": [
                "another",
                "list"
            ]
        }
    },
    "nothing": null,
    "object": {
        "currency": "USD",
        "value": 42.99
    },
    "pi": 3.141,
    "size": 8
}
PS C:\Users\hi\Desktop\projects\python_projects\ai_projects\NLP\NLP_API_VyoriusTest\test_libcurl_nlohmannJson>

*/