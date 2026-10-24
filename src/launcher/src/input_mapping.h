#pragma once

#include <map>
#include <string>
#include <vector>

namespace kyty_launcher
{

struct Settings;

// One row of the Controls page: a PS5 control, the section it belongs to, and the binding it shows.
struct ControlBinding
{
	std::string id;
	std::string label;
	std::string binding; // as shown: "Left Shift", "A / Cross", "None" when unbound
	std::string section;
	bool        section_start = false; // first row of its section: the page shows the section title
};

// One gamepad source of the fallback dropdown. XInput reports no Guide, Touchpad, paddles or Misc1 button,
// so those are only reachable by picking them here.
struct PadSourceOption
{
	std::string id;
	std::string label;
};

// What one capture poll found. Reserved/Unsupported carry the message to show and keep the capture going.
struct CaptureHit
{
	enum class Kind
	{
		None,
		Key,
		Pad,
		Cancelled,
		Reserved,
		Unsupported,
	};
	Kind        kind = Kind::None;
	std::string name;
};

// The Controls page's working copy of Settings::host_input_mapping: the two tabs, the shared touch pad
// options and the keyboard/mouse + XInput capture poll. Editing only marks a tab as custom; nothing is
// written back until SaveMapping, and RestoreDefaults puts a tab back to "not custom" (so it writes nothing).
//
// The mapping names are the Qt launcher's (inputMappingDialog.cpp): its key names and PAD_SOURCES ids are
// what the emulator's --keymap reads.
class InputMapping
{
public:
	// Reads Settings::host_input_mapping. A tab without entries of its own shows the table defaults and stays
	// not custom.
	void Load(const Settings& s);

	// Rows of one tab in table order: pad rows only for the controls that have a pad default.
	std::vector<ControlBinding> Rows(bool pad) const;
	std::string                 RowLabel(bool pad, const std::string& id) const;
	static const std::vector<PadSourceOption>& Sources();
	static std::string                         SourceLabel(const std::string& id);
	// The touch pad particle skins: ids as the emulator knows them, and their labels.
	static const std::vector<std::string>& SkinIds();
	static const std::vector<std::string>& SkinLabels();
	// False off Windows: there is no polling capture there, only the source dropdown.
	static bool CaptureAvailable();

	// Binds `binding` to `id`, first taking it from any other control of the same tab (upstream does the
	// same). An empty binding clears the control. An unknown id or binding is ignored.
	void Assign(bool pad, const std::string& id, const std::string& binding);
	void Clear(bool pad, const std::string& id);
	// The tab's table defaults; the tab counts as not custom afterwards, so saving writes nothing for it.
	void RestoreDefaults(bool pad);

	void   SetTouchOptions(double sensitivity, int skin_index, bool show_cursor);
	double Sensitivity() const { return m_sensitivity; }
	int    SkinIndex() const { return m_skin; }
	bool   ShowCursor() const { return m_show_cursor; }

	// Settings::host_input_mapping for the current state, in the upstream dialog's order (keyboard, pad,
	// then the flags).
	std::vector<std::string> Mapping() const;
	void                     SaveMapping(Settings& s) const;

	// Capture: Begin arms the poll (200 ms; anything already held is ignored until released), Poll returns
	// one hit per press, Cancel stops it. `error` says why a capture can't start.
	bool BeginCapture(bool pad, const std::string& id, std::string* error);
	void CancelCapture();
	bool Capturing() const { return m_capturing; }
	CaptureHit PollCapture();

	// "Connected: Controller 1" or "No controller connected"; off Windows it says capture is unavailable.
	std::string PadStatus() const;

private:
	const std::string* Find(bool pad, const std::string& id) const;
	void               AssignRaw(bool pad, const std::string& id, const std::string& binding);
	void               FillDefaults(bool pad);
	bool               HasRow(bool pad, const std::string& id) const;

	std::map<std::string, std::string> m_keyboard;
	std::map<std::string, std::string> m_pad;
	bool                               m_custom_keyboard = false;
	bool                               m_custom_pad      = false;
	double                             m_sensitivity     = 1.0;
	int                                m_skin            = 0;
	bool                               m_show_cursor     = true;

	// Capture state, all driven by PollCapture on the UI thread.
	bool                      m_capturing    = false;
	bool                      m_capture_pad  = false;
	std::string               m_capture_id;
	long long                 m_arm_ms       = 0; // steady clock time the poll starts reporting
	std::map<int, bool>       m_key_down;          // last seen state of every polled virtual key
	unsigned long             m_pad_buttons  = 0;
	bool                      m_pad_lt       = false;
	bool                      m_pad_rt       = false;
	int                       m_pad_index    = -1;
	bool                      m_pad_seen     = false;
};

} // namespace kyty_launcher
