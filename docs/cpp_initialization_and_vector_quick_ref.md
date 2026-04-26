# C++ Initialization and `std::vector` Quick Reference

## Variable Initialization

In modern C++, prefer initializing variables at the point of declaration.

Common forms:

```cpp
int a = 3;     // copy initialization
int b(3);      // direct initialization
int c{3};      // brace initialization
int d{};       // value initialization -> 0
```

Practical rules:

- `int x;` leaves a local variable uninitialized. Reading it is a bug.
- `int x{};` safely initializes it to zero.
- `T obj{};` is a good default habit for plain values.
- Brace initialization prevents narrowing conversions.

Example:

```cpp
int x = 3.9;   // allowed, truncates to 3
int y{3.9};    // error: narrowing conversion
```

For class types:

```cpp
std::string s1;      // default constructed, empty
std::string s2{"hi"};
```

References and `const` values usually must be initialized immediately:

```cpp
const int n = 5;
int value = 10;
int& ref = value;
```

Pointers:

```cpp
int* p{};      // null pointer
```

Good default rule:

- For simple values, use `{}`.
- Initialize immediately unless there is a real reason not to.

## `std::vector`

`std::vector<T>` is a dynamically sized contiguous array.

Example:

```cpp
std::vector<int> v;
```

Common initialization patterns:

```cpp
std::vector<int> a;            // empty
std::vector<int> b{1, 2, 3};   // elements 1, 2, 3
std::vector<int> c(5);         // 5 ints, each initialized to 0
std::vector<int> d(5, 42);     // 5 ints, each 42
```

Important distinction:

```cpp
std::vector<int> x(3, 7);  // [7, 7, 7]
std::vector<int> y{3, 7};  // [3, 7]
```

Core operations:

```cpp
v.push_back(10);     // append
v.emplace_back(20);  // construct in place
v.size();            // number of elements
v.empty();           // true if no elements
v[0];                // unchecked access
v.at(0);             // checked access, throws on bad index
v.front();           // first element
v.back();            // last element
v.clear();           // remove all elements
```

Looping:

```cpp
for (int x : v) {
    std::cout << x << '\n';
}
```

Modify elements by reference:

```cpp
for (int& x : v) {
    x *= 2;
}
```

Memory behavior:

- `std::vector` grows automatically.
- Reallocation can invalidate pointers, iterators, and references to elements.
- If you know the rough size up front, reserve capacity.

```cpp
v.reserve(100);
```

Useful mental model:

- `std::vector` is the default container when you want "a list of things".
- It is fast for appending and indexed access.
- It is not ideal for frequent insertions or removals in the middle.

Examples with your alias:

```cpp
using Vector = std::vector<float>;

Vector myVec{};                    // empty vector of floats
Vector weights{0.1f, 0.2f, 0.3f};  // list initialization
Vector zeros(10, 0.0f);            // 10 zeros
```
