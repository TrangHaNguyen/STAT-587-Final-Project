#include<retrieve_tickers.h>
#include<iostream>
#include<filesystem>
#include<sstream>

int main() {
    // Ensuring python environment exists.
    std::filesystem::path cwd = std::filesystem::current_path();
    #ifdef _WIN32
        std::filesystem::path env_dir = std::filesystem::current_path() / "PyScripts" / "env";
        std::filesystem::path env_pyt = env_dir / "Scripts" / "python.exe";
    #else
        std::filesystem::path env_dir = std::filesystem::current_path() / "PyScripts" / "env";
        std::filesystem::path env_pyt = env_dir / "bin" / "python3";
    #endif
    if (!std::filesystem::exists(env_pyt)) {
        std::cout << "Initializing virtual environment...\n";
        std::system("python -m venv PyScripts/env");
        if (std::filesystem::exists("PyScripts/requirements.txt")) {
            std::cout << "Downloading dependencies...\n";
            std::system("PyScripts/env/bin/pip install -r PyScripts/requirements.txt");
        }
        std::cout << "Finished!\n";
    }

    std::vector<std::string> TICKERS = retrieve_tickers();

    // env_pyt does not require .string() and in-place quotation marks because it contains the entire path we are looking for, 
    // whereas we are adding the current working directing (cwd) to the script of interest.
    std::stringstream ss;
    ss << env_pyt << " \"" << cwd.string() << "/PyScripts/data_manager.py\" ";
    for (int i = 0; i < TICKERS.size(); i++) ss << TICKERS[i] << " ";
    std::system(ss.str().c_str());


    return 1;
}