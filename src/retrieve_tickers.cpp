#include "retrieve_tickers.h"
#include <iostream>
#include <cpr/cpr.h>
#include <regex>
#include <vector>    

std::vector<std::string> retrieve_tickers() {
    std::string url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies";
    cpr::Response r = cpr::Get(cpr::Url{ url });

    if (r.status_code != 200) {
        std::cerr << "Failed to fetch data. Status: " << r.status_code << "\n";
        return std::vector<std::string>{""};
    }

    std::string html = r.text;
    std::regex ticker_regex(R"(<tr>\s*<td><a[^>]*>([A-Z.]+)</a>\s*</td>)");
    std::smatch match;
    std::vector<std::string> TICKERS;

    while (std::regex_search(html, match, ticker_regex)) {
        std::string TICKER = match[1];
        std::replace(TICKER.begin(), TICKER.end(), '.', '-');
        TICKERS.push_back(TICKER);
        html = match.suffix().str();
    }
    return TICKERS;
}