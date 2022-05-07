#pragma once
#include <string>
namespace qvf{
    class ComEndPoint{
        private:
            std::string ip_address;
            int port;
            int rank;
        
        public:
        ComEndPoint(int rank, std::string ip_address, int port): rank(rank), ip_address(ip_address), port(port){}
        std::string get_address(void){
            return ip_address;
        }
        int get_port(void){
            return port;
        }
        int get_rank(void){
            return rank;
        }

    };
} // namespace 