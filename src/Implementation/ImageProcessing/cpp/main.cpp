#include "Binarization.hpp"
#include "Scanner.hpp"
#include "LineSegmentation.hpp"
#include "WordSegmentation.hpp"
#include "ImageProcessing.hpp"

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <srcPath> <outPath>" << std::endl;
        return -1;
    }
    
    std::string srcPath = argv[1];
    std::string outPath = argv[2];

    ProcessImage(srcPath, outPath);
    return 0;
}
