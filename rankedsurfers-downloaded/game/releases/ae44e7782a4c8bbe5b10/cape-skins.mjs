// Only loaded by the authenticated Halloween/cape arena. Public skins and
// historical replay data are unchanged. No game physics or account mutations.
export function createCapeSkinPolicy(send) {
  let initialized = false;
  let probing = false;
  return {
    command(command) {
      return command.type === 'skin' && Number(command.skin) === 1001
        ? {...command, skin: 0}
        : command;
    },
    event(event) {
      if (event.type === 'loaded' && !event.bridgeReady && !initialized) {
        initialized = true;
        // RankedNet is now ready. Ask for the persisted selection before the
        // parent can start a round. Hide only this probe's lobby navigation.
        probing = true;
        try { send({type: 'skins'}); }
        finally { probing = false; }
      }
      if (probing && event.type === 'lobby') return false;
      if (event.type === 'skin-selected' && Number(event.skin) === 1001) {
        // Forward the real skin-selected confirmation from this selection,
        // never pretend the native game has switched before it actually does.
        send({type: 'skin', skin: 0});
        return false;
      }
      return true;
    }
  };
}
