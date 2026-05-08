
#============================================================================= #type: ignore  # noqa E501
# Copyright © 2025 NaturalPoint, Inc. All Rights Reserved.
#
# THIS SOFTWARE IS GOVERNED BY THE OPTITRACK PLUGINS EULA AVAILABLE AT https://www.optitrack.com/about/legal/eula.html #type: ignore  # noqa E501
# AND/OR FOR DOWNLOAD WITH THE APPLICABLE SOFTWARE FILE(S) (“PLUGINS EULA”). BY DOWNLOADING, INSTALLING, ACTIVATING #type: ignore  # noqa E501
# AND/OR OTHERWISE USING THE SOFTWARE, YOU ARE AGREEING THAT YOU HAVE READ, AND THAT YOU AGREE TO COMPLY WITH AND ARE #type: ignore  # noqa E501
# BOUND BY, THE PLUGINS EULA AND ALL APPLICABLE LAWS AND REGULATIONS. IF YOU DO NOT AGREE TO BE BOUND BY THE PLUGINS #type: ignore  # noqa E501
# EULA, THEN YOU MAY NOT DOWNLOAD, INSTALL, ACTIVATE OR OTHERWISE USE THE SOFTWARE AND YOU MUST PROMPTLY DELETE OR #type: ignore  # noqa E501
# RETURN IT. IF YOU ARE DOWNLOADING, INSTALLING, ACTIVATING AND/OR OTHERWISE USING THE SOFTWARE ON BEHALF OF AN ENTITY, #type: ignore  # noqa E501
# THEN BY DOING SO YOU REPRESENT AND WARRANT THAT YOU HAVE THE APPROPRIATE AUTHORITY TO ACCEPT THE PLUGINS EULA ON #type: ignore  # noqa E501
# BEHALF OF SUCH ENTITY. See license file in root directory for additional governing terms and information. #type: ignore  # noqa E501
#============================================================================= #type: ignore  # noqa E501


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish
# a connection and receive data via that NatNet connection
# to decode it using the NatNetClientLibrary.

import sys
import time
import socket
from NatNetClient import NatNetClient
import DataDescriptions
import MoCapData

# --- UDP Configuration (must match UDPMarkerReceiver.listenPort in Unity) ---
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5005

# Global UDP socket for forwarding marker data to Unity
_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Persistence filter ---------------------------------------------------
# OptiTrack streams every visible marker, including the eight infrared
# markers attached to the VR headset. Those markers are detected
# intermittently and flicker on for one or two frames at a time before
# disappearing again. The body markers we care about (shoulder, elbow,
# hand — three markers) are continuously tracked across many frames.
#
# We can therefore separate the two populations purely by temporal
# persistence: a marker is forwarded only if a marker at roughly the
# same position appeared in every one of the previous PERSISTENCE_FRAMES-1
# frames. The flicker markers never accumulate enough history and are
# silently dropped before they leave this process.
PERSISTENCE_FRAMES = 3      # current frame + (this-1) previous frames
POSITION_TOL_M     = 0.05   # 5 cm tolerance when matching across frames
_marker_history: list[list[tuple[float, float, float]]] = []


def _has_neighbour(pos, candidates, tol):
    px, py, pz = pos
    tol_sq = tol * tol
    for cx, cy, cz in candidates:
        dx, dy, dz = px - cx, py - cy, pz - cz
        if dx * dx + dy * dy + dz * dz < tol_sq:
            return True
    return False


def _filter_persistent(current_positions):
    """Return the subset of current_positions for which a nearby marker
    appeared in every frame of the persistence history. Updates the
    history afterwards so the call has a single side effect."""
    if len(_marker_history) < PERSISTENCE_FRAMES - 1:
        kept = []
    else:
        kept = [
            pos for pos in current_positions
            if all(_has_neighbour(pos, frame, POSITION_TOL_M)
                   for frame in _marker_history)
        ]
    _marker_history.append(list(current_positions))
    while len(_marker_history) > PERSISTENCE_FRAMES - 1:
        _marker_history.pop(0)
    return kept

# This is a callback function that gets connected to the NatNet client
# and called once per mocap frame.


def receive_new_frame(data_dict):
    order_list = ["frameNumber", "markerSetCount", "unlabeledMarkersCount", #type: ignore  # noqa F841
                  "rigidBodyCount", "skeletonCount", "labeledMarkerCount",
                  "timecode", "timecodeSub", "timestamp", "isRecording",
                  "trackedModelsChanged"]
    dump_args = False
    if dump_args is True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "= "
            if key in data_dict:
                out_string += data_dict[key] + " "
            out_string += "/"
        print(out_string)


def receive_new_frame_with_data(data_dict):
    order_list = ["frameNumber", "markerSetCount", "unlabeledMarkersCount", #type: ignore  # noqa F841
                  "rigidBodyCount", "skeletonCount", "labeledMarkerCount",
                  "timecode", "timecodeSub", "timestamp", "isRecording",
                  "trackedModelsChanged", "offset", "mocap_data"]
    dump_args = True
    if dump_args is True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "= "
            if key in data_dict:
                out_string += str(data_dict[key]) + " "
            out_string += "/"
        print(out_string)

    # Forward labeled marker data to Unity via UDP, after dropping flicker
    # markers (see _filter_persistent for the rationale).
    frame_number = data_dict.get("frameNumber", 0)
    mocap_data = data_dict.get("mocap_data", None)
    if mocap_data is None:
        return
    labeled_marker_data = mocap_data.labeled_marker_data
    if labeled_marker_data is None:
        return
    marker_list = labeled_marker_data.labeled_marker_list
    if len(marker_list) == 0:
        # Still update the persistence buffer so it ages out stale frames.
        _filter_persistent([])
        return

    current_positions = [tuple(m.pos) for m in marker_list]
    kept_positions = _filter_persistent(current_positions)
    if not kept_positions:
        return

    # Match each kept position back to its source marker so we preserve the
    # original IDs in the UDP message (the receiver still uses them as
    # dictionary keys when assigning marker spheres).
    kept_set = {(round(p[0], 6), round(p[1], 6), round(p[2], 6))
                for p in kept_positions}
    forwarded = []
    for marker in marker_list:
        key = (round(marker.pos[0], 6), round(marker.pos[1], 6),
               round(marker.pos[2], 6))
        if key in kept_set:
            forwarded.append(marker)

    if not forwarded:
        return

    udp_parts = [str(frame_number), str(len(forwarded))]
    for marker in forwarded:
        marker_id = marker.id_num & 0x0000FFFF
        pos = marker.pos
        udp_parts.append(f"{marker_id},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}")
    message = ";".join(udp_parts)
    _udp_socket.sendto(message.encode("utf-8"), (UNITY_IP, UNITY_PORT))


# This is a callback function that gets connected to the NatNet client.
# It is called once per rigid body per frame.
def receive_rigid_body_frame(new_id, position, rotation):
    pass
    # print("Received frame for rigid body", new_id)
    # print("Received frame for rigid body", new_id," ",position," ",rotation)


def add_lists(totals, totals_tmp):
    totals[0] += totals_tmp[0]
    totals[1] += totals_tmp[1]
    totals[2] += totals_tmp[2]
    return totals


def print_configuration(natnet_client):
    natnet_client.refresh_configuration()
    print("Connection Configuration:")
    print("  Client:          %s" % natnet_client.local_ip_address)
    print("  Server:          %s" % natnet_client.server_ip_address)
    print("  Command Port:    %d" % natnet_client.command_port)
    print("  Data Port:       %d" % natnet_client.data_port)

    changeBitstreamString = "  Can Change Bitstream Version = "
    if natnet_client.use_multicast:
        print("  Using Multicast")
        print("  Multicast Group: %s" % natnet_client.multicast_address)
        changeBitstreamString += "false"
    else:
        print("  Using Unicast")
        changeBitstreamString += "true"

    # NatNet Server Info
    application_name = natnet_client.get_application_name()
    nat_net_requested_version = natnet_client.get_nat_net_requested_version()
    nat_net_version_server = natnet_client.get_nat_net_version_server()
    server_version = natnet_client.get_server_version()

    print("  NatNet Server Info")
    print("    Application Name %s" % (application_name))
    print("    MotiveVersion  %d %d %d %d" % (server_version[0], server_version[1], server_version[2], server_version[3]))  #type: ignore  # noqa F501
    print("    NatNetVersion  %d %d %d %d" % (nat_net_version_server[0], nat_net_version_server[1], nat_net_version_server[2], nat_net_version_server[3])) #type: ignore  # noqa F501
    print("  NatNet Bitstream Requested")
    print("    NatNetVersion  %d %d %d %d" % (nat_net_requested_version[0], nat_net_requested_version[1], #type: ignore  # noqa F501
                                              nat_net_requested_version[2], nat_net_requested_version[3])) #type: ignore  # noqa F501

    print(changeBitstreamString)
    # print("command_socket = %s" % (str(natnet_client.command_socket)))
    # print("data_socket    = %s" % (str(natnet_client.data_socket)))
    print("  PythonVersion    %s" % (sys.version))


def print_commands(can_change_bitstream):
    outstring = "Commands:\n"
    outstring += "Return Data from Motive\n"
    outstring += "  s  send data descriptions\n"
    outstring += "  r  resume/start frame playback\n"
    outstring += "  p  pause frame playback\n"
    outstring += "     pause may require several seconds\n"
    outstring += "     depending on the frame data size\n"
    outstring += "Change Working Range\n"
    outstring += "  o  reset Working Range to: start/current/end frame 0/0/end of take\n" #type: ignore  # noqa F501
    outstring += "  w  set Working Range to: start/current/end frame 1/100/1500\n" #type: ignore  # noqa F501
    outstring += "Return Data Display Modes\n"
    outstring += "  j  print_level = 0 supress data description and mocap frame data\n" #type: ignore  # noqa F501
    outstring += "  k  print_level = 1 show data description and mocap frame data\n" #type: ignore  # noqa F501
    outstring += "  l  print_level = 20 show data description and every 20th mocap frame data\n" #type: ignore  # noqa F501
    outstring += "Change NatNet data stream version (Unicast only)\n"
    outstring += "  3  Request NatNet 3.1 data stream (Unicast only)\n"
    outstring += "  4  Request NatNet 4.1 data stream (Unicast only)\n"
    outstring += "General\n"
    outstring += "  t  data structures self test (no motive/server interaction)\n" #type: ignore  # noqa F501
    outstring += "  c  print configuration\n"
    outstring += "  h  print commands\n"
    outstring += "  q  quit\n"
    outstring += "\n"
    outstring += "NOTE: Motive frame playback will respond differently in\n"
    outstring += "       Endpoint, Loop, and Bounce playback modes.\n"
    outstring += "\n"
    outstring += "EXAMPLE: PacketClient [serverIP [ clientIP [ Multicast/Unicast]]]\n" #type: ignore  # noqa F501
    outstring += "         PacketClient \"192.168.10.14\" \"192.168.10.14\" Multicast\n" #type: ignore  # noqa F501
    outstring += "         PacketClient \"127.0.0.1\" \"127.0.0.1\" u\n"
    outstring += "\n"
    print(outstring)


def request_data_descriptions(s_client):
    # Request the model definitions
    s_client.send_request(s_client.command_socket, s_client.NAT_REQUEST_MODELDEF, "",  (s_client.server_ip_address, s_client.command_port)) #type: ignore  # noqa F501


def test_classes():
    totals = [0, 0, 0]
    print("Test Data Description Classes")
    totals_tmp = DataDescriptions.test_all()
    totals = add_lists(totals, totals_tmp)
    print("")
    print("Test MoCap Frame Classes")
    totals_tmp = MoCapData.test_all()
    totals = add_lists(totals, totals_tmp)
    print("")
    print("All Tests totals")
    print("--------------------")
    print("[PASS] Count = %3.1d" % totals[0])
    print("[FAIL] Count = %3.1d" % totals[1])
    print("[SKIP] Count = %3.1d" % totals[2])


def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len = len(arg_list)
    if arg_list_len > 1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len > 2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len > 3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False
        if arg_list_len > 4:
            args_dict["stream_type"] = arg_list[4]
    return args_dict


if __name__ == "__main__":

    optionsDict = {}
    optionsDict["clientAddress"] = "127.0.0.1"
    optionsDict["serverAddress"] = "127.0.0.1"
    optionsDict["use_multicast"] = None
    optionsDict["stream_type"] = None
    stream_type_arg = None

    # This will create a new NatNet client
    optionsDict = my_parse_args(sys.argv, optionsDict)
    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])

    # Streaming client configuration.
    # Calls RB handler on emulator for data transmission.
    streaming_client.new_frame_listener = receive_new_frame
    streaming_client.new_frame_with_data_listener = receive_new_frame_with_data
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # print instructions
    print("NatNet Python Client 4.4")
    print(f"UDP forwarding to Unity: {UNITY_IP}:{UNITY_PORT}\n")

    # Select Multicast or Unicast
    cast_choice = input("Select 0 for multicast and 1 for unicast: ")
    cast_choice = int(cast_choice)
    while cast_choice != 0 and cast_choice != 1:
        cast_choice = input("Invalid option. Select 0 for multicast or 1 for unicast: ") #type: ignore  # noqa F501
        cast_choice = int(cast_choice)
    # establishes multicast or unicast
    if cast_choice == 0:
        optionsDict["use_multicast"] = True
    else:
        optionsDict["use_multicast"] = False
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # allows user to set local address:
    client_addr_choice = input("Client Address (127.0.0.1): ")
    if client_addr_choice != "":
        streaming_client.set_client_address(client_addr_choice)

    # allows user to set remote address
    server_addr_choice = input("Server Address (127.0.0.1): ")
    if server_addr_choice != "":
        streaming_client.set_server_address(server_addr_choice)

    # select datastream preference
    stream_choice = None
    while stream_choice != 'd' and stream_choice != 'c':
        stream_choice = input("Select d for datastream and c for command stream: ") #type: ignore  # noqa F501
    optionsDict["stream_type"] = stream_choice

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run(optionsDict["stream_type"])
    if not is_running:
        print("ERROR: Could not start streaming client.")
        try:
            sys.exit(1)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    is_looping = True
    time.sleep(1)
    if streaming_client.connected() is False:
        print("ERROR: Could not connect properly.  Check that Motive streaming is on.") #type: ignore  # noqa F501
        try:
            sys.exit(2)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    print_configuration(streaming_client)
    print("\n")
    print_commands(streaming_client.can_change_bitstream_version())

    while is_looping:
        inchars = input('Enter command or (\'h\' for list of commands)\n')
        if len(inchars) > 0:
            c1 = inchars[0].lower()
            if c1 == 'h':
                print_commands(streaming_client.can_change_bitstream_version())
            elif c1 == 'c':
                print_configuration(streaming_client)
            elif c1 == 's':
                request_data_descriptions(streaming_client)
                time.sleep(1)
            elif (c1 == '3') or (c1 == '4'):
                if streaming_client.can_change_bitstream_version():
                    tmp_major = 4
                    tmp_minor = 2
                    if (c1 == '3'):
                        tmp_major = 3
                        tmp_minor = 1
                    return_code = streaming_client.set_nat_net_version(tmp_major, tmp_minor) #type: ignore  # noqa F501
                    time.sleep(1)
                    if return_code == -1:
                        print("Could not change bitstream version to %d.%d" % (tmp_major, tmp_minor)) #type: ignore  # noqa F501
                    else:
                        print("Bitstream version at %d.%d" % (tmp_major, tmp_minor)) #type: ignore  # noqa F501
                else:
                    print("Can only change bitstream in Unicast Mode")

            elif c1 == 'p':
                sz_command = "TimelineStop"
                return_code = streaming_client.send_command(sz_command)
                time.sleep(1)
                print("Command: %s - return_code: %d" % (sz_command, return_code)) #type: ignore  # noqa F501
            elif c1 == 'r':
                sz_command = "TimelinePlay"
                return_code = streaming_client.send_command(sz_command)
                print("Command: %s - return_code: %d" % (sz_command, return_code)) #type: ignore  # noqa F501
            elif c1 == 'o':
                tmpCommands = ["TimelinePlay",
                               "TimelineStop",
                               "SetPlaybackStartFrame,0",
                               "SetPlaybackStopFrame,1000000",
                               "SetPlaybackLooping,0",
                               "SetPlaybackCurrentFrame,0",
                               "TimelineStop"]
                for sz_command in tmpCommands:
                    return_code = streaming_client.send_command(sz_command)
                    print("Command: %s - return_code: %d" % (sz_command, return_code)) #type: ignore  # noqa F501
                time.sleep(1)
            elif c1 == 'w':
                tmp_commands = ["TimelinePlay",
                                "TimelineStop",
                                "SetPlaybackStartFrame,1",
                                "SetPlaybackStopFrame,1500",
                                "SetPlaybackLooping,0",
                                "SetPlaybackCurrentFrame,100",
                                "TimelineStop"]
                for sz_command in tmp_commands:
                    return_code = streaming_client.send_command(sz_command)
                    print("Command: %s - return_code: %d" % (sz_command, return_code)) #type: ignore  # noqa F501
                time.sleep(1)
            elif c1 == 't':
                test_classes()

            elif c1 == 'j':
                streaming_client.set_print_level(0)
                print("Showing only received frame numbers and supressing data descriptions") #type: ignore  # noqa F501
            elif c1 == 'k':
                streaming_client.set_print_level(1)
                print("Showing every received frame")

            elif c1 == 'l':
                print_level = streaming_client.set_print_level(20)
                print_level_mod = print_level % 100
                if (print_level == 0):
                    print("Showing only received frame numbers and supressing data descriptions") #type: ignore  # noqa F501
                elif (print_level == 1):
                    print("Showing every frame")
                elif (print_level_mod == 1):
                    print("Showing every %dst frame" % print_level)
                elif (print_level_mod == 2):
                    print("Showing every %dnd frame" % print_level)
                elif (print_level == 3):
                    print("Showing every %drd frame" % print_level)
                else:
                    print("Showing every %dth frame" % print_level)

            elif c1 == 'q':
                is_looping = False
                streaming_client.shutdown()
                break
            else:
                print("Error: Command %s not recognized" % c1)
            print("Ready...\n")
    _udp_socket.close()
    print("exiting")
