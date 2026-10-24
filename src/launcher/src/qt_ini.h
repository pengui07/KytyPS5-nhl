#pragma once

#include <map>
#include <string>
#include <vector>

namespace kyty_launcher {

// One value of a QSettings IniFormat file (the Qt launcher's Kyty.ini).
struct QtIniValue
{
	std::string              text;            // unescaped; a list's elements joined with ", "
	std::vector<std::string> list;            // the comma-separated elements (one element for a plain value)
	bool                     is_list = false; // the value had unquoted commas
	bool                     special = false; // @Invalid(), @ByteArray(...), @Variant(...), ...: Qt types with no text form
};

struct QtIni
{
	// "Group/key" (sub-groups and array entries: "GameConfigurations/1/name"); keys outside any group have no prefix.
	std::map<std::string, QtIniValue> values;
	int                               lines_ignored = 0; // lines that were neither [Section], key=value nor a comment

	const QtIniValue* find(const std::string& key) const;
	// A usable (not special) value as text; false when absent or special.
	bool get(const std::string& key, std::string& out) const;
	// A usable value as a string list (QSettings stores a one-element list as a plain string); false when absent or special.
	bool get_list(const std::string& key, std::vector<std::string>& out) const;
	// The keys directly under `group` ("GameConfigurations/1" -> "name", "game_path", ...), without the prefix.
	std::vector<std::string> keys_in(const std::string& group) const;
};

// Reads QSettings INI text: [Group] sections, key=value, comments with ';', QSettings escapes and quoting, "%XX" key
// characters, comma-separated lists. Tolerates a BOM and CRLF. False (with `error`) when the text holds no settings.
bool parse_qt_ini(const std::string& text, QtIni& out, std::string* error = nullptr);
bool read_qt_ini(const std::string& path, QtIni& out, std::string* error = nullptr);

} // namespace kyty_launcher
