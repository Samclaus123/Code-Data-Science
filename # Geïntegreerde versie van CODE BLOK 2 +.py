# === GEUPDATE CODE BLOK 3 (vervang je oude blok 3 hiermee) ===
import os
import glob
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, write_raw_bids

def rawtobids_with_events(raw_dir, bids_dir, subjects, event_id):
    """
    Converteer alle .bdf bestanden in raw_dir naar BIDS in bids_dir.
    Voor elk bestand:
      - zet opgegeven kanaaltypes (misc, eog)
      - zet montage biosemi64
      - vind STIM-events in raw
      - lees corresponderende *_events.tsv (zelfde basename + '_events.tsv')
      - filter phase == "2" en map (state, frequency) naar de juiste event_id
      - zet de derde kolom van de events-array naar die nieuwe codes (alleen voor de fase2-triggers)
      - schrijf naar BIDS met write_raw_bids(..., events_data=events_array, event_id=event_id)
    """
    bdf_files = sorted(glob.glob(os.path.join(raw_dir, "*.bdf")))
    print("Gevonden .bdf bestanden:", bdf_files)

    for i, bdf_file in enumerate(bdf_files):
        print("\nVerwerken:", bdf_file)
        # -- lees raw --
        raw = mne.io.read_raw_bdf(bdf_file, preload=True)

        # -- zet kanaaltypes zoals gevraagd --
        misc = ['M1', 'M2', 'EXG7', 'EXG8']
        raw.set_channel_types({ch: 'misc' for ch in misc if ch in raw.ch_names})

        eog = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        raw.set_channel_types({ch: 'eog' for ch in eog if ch in raw.ch_names})

        # -- montage --
        raw.set_montage('biosemi64', on_missing='ignore')

        # -- vind events in raw (STIM/trigger-kanaal). Gebruik veilige find_events call --
        try:
            all_events = mne.find_events(raw, shortest_event=1, verbose=False)
        except Exception as exc:
            print(f"  Kan events niet vinden met mne.find_events() voor {bdf_file}: {exc}")
            all_events = np.empty((0, 3), dtype=int)

        # Filter originele triggers: we verwachten tijdens stimuli trigger value == 2
        stim_mask = (all_events[:, 2] == 2) if all_events.size else np.array([], dtype=bool)
        stim_events = all_events[stim_mask] if stim_mask.size else np.empty((0, 3), dtype=int)
        print(f"  Aantal gevonden stimulus-triggers (waarde==2): {len(stim_events)}")

        # -- zoek corresponderend events.tsv bestand --
        base_no_ext = os.path.splitext(bdf_file)[0]
        events_tsv_path = base_no_ext + "_events.tsv"
        if not os.path.exists(events_tsv_path):
            print(f"  Waarschuwing: events file niet gevonden: {events_tsv_path}. Schrijf BIDS zonder aangepaste events.")
            events_array_to_write = None
        else:
            # -- lees events.tsv en filter phase == "2" --
            df = pd.read_csv(events_tsv_path, sep="\t", dtype=str)
            # veilig strippen en vergelijken
            if "phase" not in df.columns or "state" not in df.columns or "frequency" not in df.columns:
                print(f"  Waarschuwing: verwachte kolommen (phase,state,frequency) ontbreken in {events_tsv_path}.")
                filtered = pd.DataFrame()
            else:
                filtered = df[df["phase"].str.strip() == "2"].copy()
            print(f"  Aantal events in {os.path.basename(events_tsv_path)} met phase==2: {len(filtered)}")

            # -- map filtered rows naar event codes volgens event_id --
            new_event_codes = []
            if not filtered.empty:
                # zet veilig om naar floats/integers waar nodig
                # (we volgen hetzelfde logic als in jouw oorspronkelijke code)
                for st_str, fq_str in zip(filtered["state"], filtered["frequency"]):
                    try:
                        st = float(str(st_str).strip())
                    except:
                        st = np.nan
                    try:
                        fq = float(str(fq_str).strip())
                    except:
                        fq = np.nan

                    if st == 1:  # reeks A
                        if fq == 2000:
                            new_event_codes.append(event_id["A/standard"])
                        else:
                            new_event_codes.append(event_id["A/oddball"])
                    elif st == -1:  # reeks B
                        if fq == 1000:
                            new_event_codes.append(event_id["B/standard"])
                        else:
                            new_event_codes.append(event_id["B/oddball"])
                    else:
                        new_event_codes.append(0)  # ongedefinieerd -> code 0 (volgt jouw eerdere aanpak)

            # -- combineer de gevonden stim_events met de nieuw gemapte codes --
            n_triggers = len(stim_events)
            n_filtered = len(new_event_codes)

            if n_triggers == 0:
                print("  Geen stim-events gevonden in raw; geen events meegeven aan write_raw_bids.")
                events_array_to_write = None
            else:
                if n_triggers != n_filtered:
                    print(f"  Let op: aantal gevonden stim-events ({n_triggers}) ≠ aantal gefilterde phase2-rows ({n_filtered}).")
                    print("  Ik zal de arrays afstemmen op de minimale lengte (geen fouten raise).")
                n_use = min(n_triggers, n_filtered)
                if n_use == 0:
                    events_array_to_write = None
                else:
                    events_array_to_write = stim_events[:n_use].copy()
                    events_array_to_write[:, 2] = np.array(new_event_codes[:n_use], dtype=int)
                    print(f"  Events array klaargezet met {n_use} events (kolom 3 aangepast).")

        # -- BIDSPath maken (subject index i -> subjects[i]) --
        bids_path = BIDSPath(subject=subjects[i] if i < len(subjects) else subjects[-1],
                             root=bids_dir,
                             datatype='eeg',
                             extension='.bdf',
                             suffix='eeg',
                             task='experiment')

        # -- schrijf naar BIDS (events_data indien beschikbaar) --
        if events_array_to_write is None:
            write_raw_bids(raw, bids_path=bids_path, overwrite=True, allow_preload=True, format='EDF')
            print(f"  {os.path.basename(bdf_file)} geschreven naar BIDS (zonder custom events).")
        else:
            write_raw_bids(raw,
                           bids_path=bids_path,
                           events_data=events_array_to_write,
                           event_id=event_id,
                           overwrite=True,
                           allow_preload=True,
                           format='EDF')
            print(f"  {os.path.basename(bdf_file)} geschreven naar BIDS (met aangepaste events).")

    print("\nKlaar met alle bestanden.")

# --- Aanroep (vervang of gebruik deze aanroep in je notebook) ---
rawtobids_with_events(raw_dir=raw_dir, bids_dir=bids_dir, subjects=subjects, event_id=event_id)
