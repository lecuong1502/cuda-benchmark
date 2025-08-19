// File: measure the time of processing for CPU and GPU

#ifndef TIMER_H
#define TIMER_H

#include <chrono>     // For high-resolution clock
#include <string>
#include <iostream>

using namespace std;

class Timer {
private:
    string name;
    std::chrono::high_resolution_clock::time_point start;

public:
    Timer(const string& timer_name) : name(timer_name) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        cout << "Timer " << name << " | Duration time: " 
             << duration.count() << " seconds" << endl;
    }
};

#endif // TIMER_H