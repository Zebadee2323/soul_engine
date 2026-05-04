#include <afex/host.hpp>
// --------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace afex
{

namespace
{

std::string trim(std::string_view value) {
    auto first = std::size_t{0};
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first])) != 0) {
        ++first;
    }

    auto last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1])) != 0) {
        --last;
    }

    return std::string{value.substr(first, last - first)};
}

std::string unquote(std::string value) {
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') || (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }

    return value;
}

std::string strip_comment(std::string_view line) {
    auto in_single_quote = false;
    auto in_double_quote = false;

    for (auto i = std::size_t{0}; i < line.size(); ++i) {
        const auto character = line[i];
        if (character == '\'' && !in_double_quote) {
            in_single_quote = !in_single_quote;
        } else if (character == '"' && !in_single_quote) {
            in_double_quote = !in_double_quote;
        } else if (character == '#' && !in_single_quote && !in_double_quote) {
            return std::string{line.substr(0, i)};
        }
    }

    return std::string{line};
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

void add_key_value(ExtractorConfig& extractor, std::string_view key, std::string_view raw_value) {
    const auto value = trim(raw_value);
    if (key == "name") {
        extractor.name = unquote(value);
        return;
    }

    if (key == "parameters") {
        return;
    }

    extractor.parameters[std::string{key}] = std::stod(value);
}

void parse_inline_list_item(AnalysisConfig& config, std::string value) {
    auto extractor = ExtractorConfig{};
    value = trim(value);
    if (value.empty()) {
        config.extractors.push_back(std::move(extractor));
        return;
    }

    const auto separator = value.find(':');
    if (separator == std::string::npos) {
        extractor.name = unquote(value);
    } else {
        add_key_value(extractor, trim(std::string_view{value}.substr(0, separator)), std::string_view{value}.substr(separator + 1));
    }

    config.extractors.push_back(std::move(extractor));
}

}

AnalysisConfig load_analysis_config_yaml(std::string_view config_file_path) {
    auto stream = std::ifstream{std::string{config_file_path}};
    if (!stream) {
        throw std::runtime_error("Unable to open afex YAML config: " + std::string{config_file_path});
    }

    auto config = AnalysisConfig{};
    auto in_extractors = false;
    auto in_parameters = false;
    auto line = std::string{};

    while (std::getline(stream, line)) {
        const auto without_comment = strip_comment(line);
        const auto text = trim(without_comment);
        if (text.empty()) {
            continue;
        }

        if (text == "extractors:") {
            in_extractors = true;
            in_parameters = false;
            continue;
        }

        if (!in_extractors) {
            const auto separator = text.find(':');
            if (separator == std::string::npos) {
                continue;
            }

            const auto key = trim(std::string_view{text}.substr(0, separator));
            const auto value = trim(std::string_view{text}.substr(separator + 1));
            if (key == "audio_file" || key == "audio_file_path") {
                config.audio_file_path = unquote(value);
            }
            continue;
        }

        if (starts_with(text, "-")) {
            in_parameters = false;
            parse_inline_list_item(config, trim(std::string_view{text}.substr(1)));
            continue;
        }

        if (config.extractors.empty()) {
            throw std::runtime_error("YAML extractor property appears before any extractor list item.");
        }

        if (text == "parameters:") {
            in_parameters = true;
            continue;
        }

        const auto separator = text.find(':');
        if (separator == std::string::npos) {
            continue;
        }

        const auto key = trim(std::string_view{text}.substr(0, separator));
        const auto value = trim(std::string_view{text}.substr(separator + 1));
        auto& extractor = config.extractors.back();
        if (in_parameters) {
            extractor.parameters[key] = std::stod(value);
        } else {
            add_key_value(extractor, key, value);
        }
    }

    for (const auto& extractor : config.extractors) {
        if (extractor.name.empty()) {
            throw std::runtime_error("Every configured extractor must include a name.");
        }
    }

    return config;
}

Status load_analysis_config_yaml(std::string_view config_file_path, AnalysisConfig& config) {
#if AFEX_ENABLE_EXCEPTIONS
    try {
        config = load_analysis_config_yaml(config_file_path);
        return {};
    } catch (const std::invalid_argument& error) {
        return Status{.error = AfexError::InvalidArgument, .message = error.what()};
    } catch (const std::runtime_error& error) {
        return Status{.error = AfexError::ParseFailed, .message = error.what()};
    }
#else
    static_cast<void>(config_file_path);
    static_cast<void>(config);
    return Status{.error = AfexError::ParseFailed, .message = "Host YAML loading requires AFEX_ENABLE_EXCEPTIONS in this build."};
#endif
}

}
