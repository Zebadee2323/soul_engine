# `std::vector` Cheatsheet

`std::vector<T>` is a dynamically sized contiguous array.

It is usually the default C++ container when you want:
- a growable list
- fast indexed access
- good cache-friendly layout

## Include

```cpp
#include <vector>
```

## Basic Declaration

```cpp
std::vector<int> a;
std::vector<float> b;
```

## Common Initialization Patterns

```cpp
std::vector<int> a;           // empty
std::vector<int> b{};         // empty
std::vector<int> c{1, 2, 3};  // elements: 1, 2, 3
std::vector<int> d(5);        // 5 zeros
std::vector<int> e(5, 42);    // 5 elements, all 42
```

Important distinction:

```cpp
std::vector<int> x(3, 7);  // [7, 7, 7]
std::vector<int> y{3, 7};  // [3, 7]
```

## Add Elements

```cpp
std::vector<int> v;

v.push_back(10);      // append an existing value
v.emplace_back(20);   // construct element in place
```

For simple types like `int`, `push_back` and `emplace_back` are usually equivalent in practice.

The difference matters more for class/struct types:

```cpp
MlpLayer layer{dense_layer, activation};

v.push_back(layer);  // copy/move an existing MlpLayer into the vector
```

```cpp
v.emplace_back(dense_layer, activation);  // build the MlpLayer directly inside the vector
```

Rule of thumb:
- use `push_back(x)` when you already have the object
- use `emplace_back(args...)` when you want the vector to construct the object from arguments

## Access Elements

```cpp
v[0];        // unchecked
v.at(0);     // bounds-checked, throws on bad index
v.front();   // first element
v.back();    // last element
```

Use `at()` when safety matters more than speed.

## Useful Queries

```cpp
v.size();      // number of elements
v.empty();     // true if size is 0
v.capacity();  // current allocated capacity
```

## Looping

Read-only:

```cpp
for (const int value : v) {
    std::cout << value << '\n';
}
```

Modify in place:

```cpp
for (int& value : v) {
    value *= 2;
}
```

Index-based:

```cpp
for (std::size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i] << '\n';
}
```

## Resize and Reserve

```cpp
v.resize(10);    // change size to 10
v.reserve(100);  // pre-allocate space for 100 elements
```

Difference:
- `resize(n)` changes how many elements the vector contains
- `reserve(n)` changes only capacity, not size

Example:

```cpp
std::vector<int> v;
v.reserve(100);

for (int i = 0; i < 100; ++i) {
    v.push_back(i);
}
```

This can avoid repeated reallocations.

## Allocate Storage Without Initializing Elements

If you want space for `x` elements but do **not** want to initialize `x`
elements yet, use `reserve(x)`, not `resize(x)`:

```cpp
std::vector<float> v;
v.reserve(x);  // allocates capacity for x floats, but v.size() is still 0

// Later, construct only the elements you actually need:
v.emplace_back(1.0f);
v.emplace_back(2.0f);
```

Important:

```cpp
std::vector<float> v;
v.reserve(x);

// v[0] = 1.0f;  // wrong: no element exists yet
```

`reserve(x)` may allocate internal memory, but the vector still contains zero
live elements. You must add elements with `push_back()` / `emplace_back()` /
`insert()`.

If you need an array of size `x` whose memory is intentionally uninitialized,
`std::vector` is usually the wrong abstraction. `std::vector<T>(x)` and
`resize(x)` create `x` real `T` objects, so they initialize/construct them.

For raw uninitialized storage, prefer a lower-level tool, for example:

```cpp
#include <memory>

// C++20: array storage for trivial/buffer-like data, not initialized.
auto data = std::make_unique_for_overwrite<float[]>(x);
```

Or use an allocator directly:

```cpp
#include <memory>

std::allocator<float> alloc;

float* data = alloc.allocate(x);  // raw uninitialized storage

// construct elements manually before reading them
std::construct_at(data + 0, 1.0f);

std::destroy_at(data + 0);
alloc.deallocate(data, x);
```

Only use raw allocator storage when you really need manual lifetime control.

## Remove Elements

```cpp
v.pop_back();   // remove last element
v.clear();      // remove all elements
```

Erase one element:

```cpp
v.erase(v.begin() + 2);
```

Erase a range:

```cpp
v.erase(v.begin(), v.begin() + 3);
```

## Insert Elements

```cpp
v.insert(v.begin() + 1, 99);           // insert one value
v.insert(v.end(), {7, 8, 9});          // insert several values
```

Note: inserting or erasing in the middle is usually expensive because elements must shift.

## Pass to Functions

Read-only:

```cpp
void print(const std::vector<int>& v);
```

Modify the original:

```cpp
void double_values(std::vector<int>& v);
```

Make a copy:

```cpp
void store(std::vector<int> v);
```

Rule of thumb:
- use `const std::vector<T>&` for read-only input
- use `std::vector<T>&` when modifying
- pass by value only when you want a copy

## Common Pitfalls

### 1. `[]` does not check bounds

```cpp
v[100];   // undefined behavior if out of range
```

### 2. `reserve()` does not create elements

```cpp
std::vector<int> v;
v.reserve(10);

// v[0] is still invalid here
```

### 3. Reallocation can invalidate references and pointers

```cpp
std::vector<int> v{1, 2, 3};
int* p = &v[0];

v.push_back(4);  // may reallocate
// p may now be invalid
```

### 4. Middle insert/erase is not cheap

`std::vector` is best when you mostly:
- append at the end
- iterate
- access by index

## Handy Example

```cpp
#include <iostream>
#include <vector>

int main()
{
    std::vector<float> values{1.0f, 2.0f, 3.0f};
    values.push_back(4.0f);
    values.reserve(16);

    for (float& value : values) {
        value *= 2.0f;
    }

    for (const float value : values) {
        std::cout << value << '\n';
    }
}
```

## Quick Rules

- Use `std::vector` by default for a sequence of values.
- Use `{}` when listing elements.
- Use `(count, value)` when creating repeated values.
- Use `reserve()` when you know roughly how many elements you will append.
- Prefer `const std::vector<T>&` for read-only function parameters.
- Be careful with pointers/references after `push_back()`.
