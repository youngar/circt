#include <iostream>
int main(int argc, char *argv[]) {
    std::cerr << "server running\n";
    
    while (true) {
        unsigned char input;
        std::cin >> input;
        std::cerr << input;
    }

    std::cerr << "server closign\n";
    return EXIT_SUCCESS;
}
