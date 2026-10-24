# Local multiplayer controllers

Kyty supports four local player slots. The first physical gamepad belongs to the
configured Kyty user. Other gamepads receive virtual users named Player2, Player3,
and Player4 and separate pad handles, input history, motion, vibration, light bar,
and trigger effects. Keyboard input continues to control Player1.

On connection, the default colors are Player1 blue, Player2 red, Player3 green,
and Player4 pink. The game can then override each player's color through that
player's own pad handle, so controller colors follow the game's player indicators.
Each player slot remembers its latest game-set color and restores it on reconnect,
including requests made while its physical controller was disconnected. The startup
palette is used only before that player receives a game-set color. An explicitly
configured Player1 color still takes precedence for that player.

Assignments follow connection order and stay fixed until disconnect. A disconnected
secondary player logs out; the remaining users keep their identities. Extra attached
devices wait for a free slot. Reconnecting can assign a different available slot.

Controller speaker and vibration audio follow the user that opened the audio port.
When multiple wired DualSense audio endpoints cannot be associated reliably with
their controllers, the existing audio fallback applies. This does not affect
independent button, stick, or ordinary rumble input/output.

## Running

Multi-controller support is enabled automatically. Launch the game normally and
connect the controllers; secondary users appear as their devices connect.
Disconnect Player2 and confirm Player1 still works. The keyboard keeps Player1's
virtual pad connected when its physical controller is unplugged. Per-player
speaker audio can only be checked if the game opens such ports for both users.

Set `KYTY_DBG_CONTROLLERS=1` to log controller assignments, pad opens, and button
events even when the guest API log is configured as Silent.

## Automated checks

The controller, UserService, audio timing, and haptics tests cover independent
handles, histories, virtual IDs, login/logout ordering, slot reuse, and output
ownership. `controller_sdl_integration_tests` additionally uses four real SDL
virtual gamepads with the production controller and UserService implementations,
including 64 disconnect/reconnect cycles with new host device IDs.
Virtual devices do not establish compatibility with every physical controller or
whether a particular game presents a local multiplayer user-selection screen.
