#include "qt_ini.h"

#include "library.h"

#include <cctype>
#include <cstdlib>

namespace kyty_launcher {

namespace {

void AppendUtf8(std::string& out, unsigned cp)
{
	if (cp < 0x80)
	{
		out += static_cast<char>(cp);
	} else if (cp < 0x800)
	{
		out += static_cast<char>(0xC0 | (cp >> 6));
		out += static_cast<char>(0x80 | (cp & 0x3F));
	} else if (cp < 0x10000)
	{
		out += static_cast<char>(0xE0 | (cp >> 12));
		out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
		out += static_cast<char>(0x80 | (cp & 0x3F));
	} else
	{
		out += static_cast<char>(0xF0 | (cp >> 18));
		out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
		out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
		out += static_cast<char>(0x80 | (cp & 0x3F));
	}
}

int HexDigit(char c)
{
	if (c >= '0' && c <= '9')
	{
		return c - '0';
	}
	if (c >= 'a' && c <= 'f')
	{
		return c - 'a' + 10;
	}
	if (c >= 'A' && c <= 'F')
	{
		return c - 'A' + 10;
	}
	return -1;
}

std::string Trim(const std::string& s)
{
	size_t b = 0;
	size_t e = s.size();
	while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0)
	{
		b++;
	}
	while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0)
	{
		e--;
	}
	return s.substr(b, e - b);
}

// A UTF-16 unit from an escape: pairs of surrogates are joined, anything else is appended as UTF-8.
void AppendUnit(std::string& out, unsigned& high, unsigned v)
{
	if (v >= 0xD800 && v < 0xDC00)
	{
		high = v;
	} else if (v >= 0xDC00 && v < 0xE000 && high != 0)
	{
		AppendUtf8(out, 0x10000 + ((high - 0xD800) << 10) + (v - 0xDC00));
		high = 0;
	} else
	{
		AppendUtf8(out, v);
	}
}

// A key or section name: "%XX" is a byte, "%UXXXX" a UTF-16 unit, '\' separates sub-groups ('/' in the result).
std::string DecodeName(const std::string& raw)
{
	std::string out;
	unsigned    high = 0;
	for (size_t i = 0; i < raw.size(); i++)
	{
		const char c = raw[i];
		if (c == '\\')
		{
			out += '/';
		} else if (c == '%' && i + 1 < raw.size() && (raw[i + 1] == 'U' || raw[i + 1] == 'u'))
		{
			unsigned v = 0;
			size_t   j = i + 2;
			int      n = 0;
			while (j < raw.size() && n < 4 && HexDigit(raw[j]) >= 0)
			{
				v = v * 16 + static_cast<unsigned>(HexDigit(raw[j]));
				j++;
				n++;
			}
			if (n == 0)
			{
				out += c;
				continue;
			}
			AppendUnit(out, high, v);
			i = j - 1;
		} else if (c == '%' && i + 2 < raw.size() && HexDigit(raw[i + 1]) >= 0 && HexDigit(raw[i + 2]) >= 0)
		{
			out += static_cast<char>(HexDigit(raw[i + 1]) * 16 + HexDigit(raw[i + 2]));
			i += 2;
		} else
		{
			out += c;
		}
	}
	return out;
}

// The value part of a line: QSettings' unescaped string list. Elements are split at commas outside quotes.
QtIniValue ParseValue(const std::string& raw)
{
	QtIniValue  v;
	std::string s = raw;
	if (!s.empty() && s[0] == '@')
	{
		if (s.size() > 1 && s[1] == '@')
		{
			s.erase(0, 1); // "@@" is a literal '@'
		} else
		{
			v.special = true;
			return v;
		}
	}
	std::string cur;
	bool        in_quote = false;
	unsigned    high     = 0;
	for (size_t i = 0; i < s.size(); i++)
	{
		const char c = s[i];
		if (c == '"')
		{
			in_quote = !in_quote;
			continue;
		}
		if (c == ',' && !in_quote)
		{
			v.list.push_back(cur);
			cur.clear();
			v.is_list = true;
			while (i + 1 < s.size() && (s[i + 1] == ' ' || s[i + 1] == '\t'))
			{
				i++;
			}
			continue;
		}
		if (c != '\\' || i + 1 >= s.size())
		{
			cur += c;
			continue;
		}
		const char e = s[++i];
		switch (e)
		{
			case 'a': cur += '\a'; break;
			case 'b': cur += '\b'; break;
			case 'f': cur += '\f'; break;
			case 'n': cur += '\n'; break;
			case 'r': cur += '\r'; break;
			case 't': cur += '\t'; break;
			case 'v': cur += '\v'; break;
			case 'x':
			{
				unsigned cp = 0;
				int      n  = 0;
				while (i + 1 < s.size() && n < 4 && HexDigit(s[i + 1]) >= 0)
				{
					cp = cp * 16 + static_cast<unsigned>(HexDigit(s[++i]));
					n++;
				}
				if (n == 0)
				{
					cur += 'x';
				} else
				{
					AppendUnit(cur, high, cp);
				}
				break;
			}
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			{
				unsigned cp = static_cast<unsigned>(e - '0');
				int      n  = 1;
				while (i + 1 < s.size() && n < 3 && s[i + 1] >= '0' && s[i + 1] <= '7')
				{
					cp = cp * 8 + static_cast<unsigned>(s[++i] - '0');
					n++;
				}
				AppendUtf8(cur, cp);
				break;
			}
			default: cur += e; break; // \\ \" \' \? \; \, \= ...: the character itself
		}
	}
	v.list.push_back(cur);
	for (size_t i = 0; i < v.list.size(); i++)
	{
		v.text += (i == 0 ? "" : ", ") + v.list[i];
	}
	return v;
}

} // namespace

const QtIniValue* QtIni::find(const std::string& key) const
{
	auto it = values.find(key);
	return it == values.end() ? nullptr : &it->second;
}

bool QtIni::get(const std::string& key, std::string& out) const
{
	const QtIniValue* v = find(key);
	if (v == nullptr || v->special)
	{
		return false;
	}
	out = v->text;
	return true;
}

bool QtIni::get_list(const std::string& key, std::vector<std::string>& out) const
{
	const QtIniValue* v = find(key);
	if (v == nullptr || v->special)
	{
		return false;
	}
	out.clear();
	if (!v->is_list && v->text.empty())
	{
		return true; // "key=": an empty list
	}
	out = v->list;
	return true;
}

std::vector<std::string> QtIni::keys_in(const std::string& group) const
{
	std::vector<std::string> out;
	const std::string        prefix = group.empty() ? std::string() : group + "/";
	for (const auto& [k, v]: values)
	{
		if (k.compare(0, prefix.size(), prefix) == 0 && k.find('/', prefix.size()) == std::string::npos)
		{
			out.push_back(k.substr(prefix.size()));
		}
	}
	return out;
}

bool parse_qt_ini(const std::string& text_in, QtIni& out, std::string* error)
{
	auto fail = [&](const char* why)
	{
		if (error != nullptr)
		{
			*error = why;
		}
		return false;
	};
	std::string text = text_in;
	if (text.rfind("\xEF\xBB\xBF", 0) == 0)
	{
		text.erase(0, 3);
	}
	if (text.find('\0') != std::string::npos)
	{
		return fail("not a text file");
	}
	out = QtIni();

	// Physical lines; a line ending in an odd number of backslashes continues on the next one.
	std::vector<std::string> lines;
	{
		std::string cur;
		for (size_t i = 0; i <= text.size(); i++)
		{
			if (i < text.size() && text[i] != '\n' && text[i] != '\r')
			{
				cur += text[i];
				continue;
			}
			if (i < text.size() && text[i] == '\r' && i + 1 < text.size() && text[i + 1] == '\n')
			{
				i++;
			}
			size_t slashes = 0;
			while (slashes < cur.size() && cur[cur.size() - 1 - slashes] == '\\')
			{
				slashes++;
			}
			if (slashes % 2 == 1 && i < text.size())
			{
				cur.pop_back();
				continue;
			}
			lines.push_back(cur);
			cur.clear();
		}
	}

	std::string prefix; // "Group/" or ""
	int         entries  = 0;
	int         sections = 0;
	for (const auto& raw_line: lines)
	{
		const std::string line = Trim(raw_line);
		if (line.empty() || line[0] == ';')
		{
			continue;
		}
		if (line[0] == '[')
		{
			const size_t close = line.rfind(']');
			if (close == std::string::npos)
			{
				out.lines_ignored++;
				continue;
			}
			const std::string name = DecodeName(line.substr(1, close - 1));
			sections++;
			prefix = (name.empty() || name == "General" || name == "%General") ? std::string() : name + "/";
			continue;
		}
		const size_t eq = line.find('=');
		if (eq == std::string::npos)
		{
			out.lines_ignored++;
			continue;
		}
		const std::string key = DecodeName(Trim(line.substr(0, eq)));
		if (key.empty())
		{
			out.lines_ignored++;
			continue;
		}
		out.values[prefix + key] = ParseValue(Trim(line.substr(eq + 1)));
		entries++;
	}
	if (entries == 0 && sections == 0)
	{
		return fail("no settings found (not a QSettings .ini file?)");
	}
	return true;
}

bool read_qt_ini(const std::string& path, QtIni& out, std::string* error)
{
	const std::string text = read_text_file(path);
	if (text.empty())
	{
		if (error != nullptr)
		{
			*error = "cannot read the file (or it is empty)";
		}
		return false;
	}
	return parse_qt_ini(text, out, error);
}

} // namespace kyty_launcher
