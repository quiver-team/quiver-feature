#pragma once

#include <iostream>
namespace qvf{
    class Range{
        private:
            uint64_t start;
            uint64_t end;
        
        public:
            Range(uint64_t start, uint64_t end): start(start), end(end){}
            uint64_t range_start(){
                return start;
            }
            uint64_t range_end(){
                return end;
            }
    };
}