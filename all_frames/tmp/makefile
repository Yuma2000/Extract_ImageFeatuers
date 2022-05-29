
CXX	= g++
TARGET	= main
OBJS	= main.o UtilityFunctions.o
CFLAGS	= -Wall -O2 -std=c++11

.SUFFIXES: .cpp .o

$(TARGET): $(OBJS)
	$(CXX) $(CFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cpp
	$(CXX) $(CFLAGS) -c main.cpp

UtilityFunctions.o: UtilityFunctions.cpp
	$(CXX) $(CFLAGS) -c UtilityFunctions.cpp

main.o: UtilityFunctions.h 
UtilityFunction.o: UtilityFunctions.h

clean:
	rm -f *~ $(TARGET) *.o

