# vcpkg Workflow

This repo uses [vcpkg manifest mode](https://learn.microsoft.com/en-us/vcpkg/concepts/manifest-mode) for C++ dependencies.

## One-time setup

1. Install vcpkg and set `VCPKG_ROOT` to your local clone.
2. Make sure `cmake` and `ninja` are installed.

Example:

```bash
git clone https://github.com/microsoft/vcpkg.git ~/dev/vcpkg
export VCPKG_ROOT=~/dev/vcpkg
```

You can put the `export` line in your shell profile.

## Project dependencies

Dependencies are declared in the repo-root `vcpkg.json`.

Current dependencies:

- `eigen3`

The manifest also pins a `builtin-baseline` so dependency resolution is repeatable across machines.

## Configure and build

Use the checked-in CMake presets:

```bash
cmake --preset debug
cmake --build --preset debug
```

Release build:

```bash
cmake --preset release
cmake --build --preset release
```

When CMake configures with the vcpkg toolchain, vcpkg will automatically install any missing manifest dependencies into the repo-local `vcpkg_installed/` directory.

## Build selected apps or libraries

Configure once from the repo root, then build only the target you want:

```bash
cmake --preset debug
cmake --build --preset mlp-debug
```

Release:

```bash
cmake --preset release
cmake --build --preset mlp-release
```

As more apps and libraries are added, give each CMake target a matching build preset at the repo root. Keep presets at the repo root rather than inside each app folder.

## Add a new dependency

Preferred:

```bash
cd /path/to/soul_engine
vcpkg add port <port-name>
```

Or edit `vcpkg.json` manually.

After that, reconfigure:

```bash
cmake --preset debug
```

Then use normal CMake integration in your `CMakeLists.txt`:

```cmake
find_package(Eigen3 REQUIRED)
target_link_libraries(my_target PRIVATE Eigen3::Eigen)
```

## Update the baseline

To refresh the pinned registry baseline:

```bash
vcpkg x-update-baseline
```

Review the `vcpkg.json` change in git after running it.

If you update your local `vcpkg` checkout and then see manifest-resolution errors during `cmake --preset ...`, run `vcpkg x-update-baseline` in this repo and recommit the updated `builtin-baseline`.

## Machine-local overrides

If you need machine-specific preset tweaks, put them in `CMakeUserPresets.json`. That file is gitignored.
