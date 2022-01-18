from pathlib import Path

from scapy.layers.dns import DNS
from scapy.layers.inet import TCP
from scapy.packet import Padding
from scapy.utils import PcapReader

# for app identification
PREFIX_TO_APP_ID = {
    # AIM chat
    'aim_chat_3a': 0,
    'aim_chat_3b': 0,
    'aimchat1': 0,
    'aimchat2': 0,
    # Email
    'email1a': 1,
    'email1b': 1,
    'email2a': 1,
    'email2b': 1,
    # Facebook
    'facebook_audio1a': 2,
    'facebook_audio1b': 2,
    'facebook_audio2a': 2,
    'facebook_audio2b': 2,
    'facebook_audio3': 2,
    'facebook_audio4': 2,
    'facebook_chat_4a': 2,
    'facebook_chat_4b': 2,
    'facebook_video1a': 2,
    'facebook_video1b': 2,
    'facebook_video2a': 2,
    'facebook_video2b': 2,
    'facebookchat1': 2,
    'facebookchat2': 2,
    'facebookchat3': 2,
    # FTPS
    'ftps_down_1a': 3,
    'ftps_down_1b': 3,
    'ftps_up_2a': 3,
    'ftps_up_2b': 3,
    # Gmail
    'gmailchat1': 4,
    'gmailchat2': 4,
    'gmailchat3': 4,
    # Hangouts
    'hangout_chat_4b': 5,
    'hangouts_audio1a': 5,
    'hangouts_audio1b': 5,
    'hangouts_audio2a': 5,
    'hangouts_audio2b': 5,
    'hangouts_audio3': 5,
    'hangouts_audio4': 5,
    'hangouts_chat_4a': 5,
    'hangouts_video1b': 5,
    'hangouts_video2a': 5,
    'hangouts_video2b': 5,
    # ICQ
    'icq_chat_3a': 6,
    'icq_chat_3b': 6,
    'icqchat1': 6,
    'icqchat2': 6,
    # Netflix
    'netflix1': 7,
    'netflix2': 7,
    'netflix3': 7,
    'netflix4': 7,
    # SCP
    'scp1': 8,
    'scpdown1': 8,
    'scpdown2': 8,
    'scpdown3': 8,
    'scpdown4': 8,
    'scpdown5': 8,
    'scpdown6': 8,
    'scpup1': 8,
    'scpup2': 8,
    'scpup3': 8,
    'scpup5': 8,
    'scpup6': 8,
    # SFTP
    'sftp1': 9,
    'sftp_down_3a': 9,
    'sftp_down_3b': 9,
    'sftp_up_2a': 9,
    'sftp_up_2b': 9,
    'sftpdown1': 9,
    'sftpdown2': 9,
    'sftpup1': 9,
    # Skype
    'skype_audio1a': 10,
    'skype_audio1b': 10,
    'skype_audio2a': 10,
    'skype_audio2b': 10,
    'skype_audio3': 10,
    'skype_audio4': 10,
    'skype_chat1a': 10,
    'skype_chat1b': 10,
    'skype_file1': 10,
    'skype_file2': 10,
    'skype_file3': 10,
    'skype_file4': 10,
    'skype_file5': 10,
    'skype_file6': 10,
    'skype_file7': 10,
    'skype_file8': 10,
    'skype_video1a': 10,
    'skype_video1b': 10,
    'skype_video2a': 10,
    'skype_video2b': 10,
    # Spotify
    'spotify1': 11,
    'spotify2': 11,
    'spotify3': 11,
    'spotify4': 11,
    # Torrent
    'torrent01': 12,
    # Tor
    'torfacebook': 13,
    'torgoogle': 13,
    'tortwitter': 13,
    'torvimeo1': 13,
    'torvimeo2': 13,
    'torvimeo3': 13,
    'toryoutube1': 13,
    'toryoutube2': 13,
    'toryoutube3': 13,
    # Vimeo
    'vimeo1': 14,
    'vimeo2': 14,
    'vimeo3': 14,
    'vimeo4': 14,
    # Voipbuster
    'voipbuster1b': 15,
    'voipbuster2b': 15,
    'voipbuster3b': 15,
    'voipbuster_4a': 15,
    'voipbuster_4b': 15,
    # Youtube
    'youtube1': 16,
    'youtube2': 16,
    'youtube3': 16,
    'youtube4': 16,
    'youtube5': 16,
    'youtube6': 16,
    'youtubehtml5_1': 16,
}

ID_TO_APP = {
    0: 'AIM Chat',
    1: 'Email',
    2: 'Facebook',
    3: 'FTPS',
    4: 'Gmail',
    5: 'Hangouts',
    6: 'ICQ',
    7: 'Netflix',
    8: 'SCP',
    9: 'SFTP',
    10: 'Skype',
    11: 'Spotify',
    12: 'Torrent',
    13: 'Tor',
    14: 'Vimeo',
    15: 'Voipbuster',
    16: 'Youtube',
}

# for traffic identification
PREFIX_TO_TRAFFIC_ID = {
    # Chat
    'aim_chat_3a': 0,
    'aim_chat_3b': 0,
    'aimchat1': 0,
    'aimchat2': 0,
    'facebook_chat_4a': 0,
    'facebook_chat_4b': 0,
    'facebookchat1': 0,
    'facebookchat2': 0,
    'facebookchat3': 0,
    'hangout_chat_4b': 0,
    'hangouts_chat_4a': 0,
    'icq_chat_3a': 0,
    'icq_chat_3b': 0,
    'icqchat1': 0,
    'icqchat2': 0,
    'skype_chat1a': 0,
    'skype_chat1b': 0,
    # Email
    'email1a': 1,
    'email1b': 1,
    'email2a': 1,
    'email2b': 1,
    # File Transfer
    'ftps_down_1a': 2,
    'ftps_down_1b': 2,
    'ftps_up_2a': 2,
    'ftps_up_2b': 2,
    'sftp1': 2,
    'sftp_down_3a': 2,
    'sftp_down_3b': 2,
    'sftp_up_2a': 2,
    'sftp_up_2b': 2,
    'sftpdown1': 2,
    'sftpdown2': 2,
    'sftpup1': 2,
    'skype_file1': 2,
    'skype_file2': 2,
    'skype_file3': 2,
    'skype_file4': 2,
    'skype_file5': 2,
    'skype_file6': 2,
    'skype_file7': 2,
    'skype_file8': 2,
    # Streaming
    'vimeo1': 3,
    'vimeo2': 3,
    'vimeo3': 3,
    'vimeo4': 3,
    'youtube1': 3,
    'youtube2': 3,
    'youtube3': 3,
    'youtube4': 3,
    'youtube5': 3,
    'youtube6': 3,
    'youtubehtml5_1': 3,
    # Torrent
    'torrent01': 4,
    # VoIP
    'facebook_audio1a': 5,
    'facebook_audio1b': 5,
    'facebook_audio2a': 5,
    'facebook_audio2b': 5,
    'facebook_audio3': 5,
    'facebook_audio4': 5,
    'hangouts_audio1a': 5,
    'hangouts_audio1b': 5,
    'hangouts_audio2a': 5,
    'hangouts_audio2b': 5,
    'hangouts_audio3': 5,
    'hangouts_audio4': 5,
    'skype_audio1a': 5,
    'skype_audio1b': 5,
    'skype_audio2a': 5,
    'skype_audio2b': 5,
    'skype_audio3': 5,
    'skype_audio4': 5,
    # VPN: Chat
    'vpn_aim_chat1a': 6,
    'vpn_aim_chat1b': 6,
    'vpn_facebook_chat1a': 6,
    'vpn_facebook_chat1b': 6,
    'vpn_hangouts_chat1a': 6,
    'vpn_hangouts_chat1b': 6,
    'vpn_icq_chat1a': 6,
    'vpn_icq_chat1b': 6,
    'vpn_skype_chat1a': 6,
    'vpn_skype_chat1b': 6,
    # VPN: File Transfer
    'vpn_ftps_a': 7,
    'vpn_ftps_b': 7,
    'vpn_sftp_a': 7,
    'vpn_sftp_b': 7,
    'vpn_skype_files1a': 7,
    'vpn_skype_files1b': 7,
    # VPN: Email
    'vpn_email2a': 8,
    'vpn_email2b': 8,
    # VPN: Streaming
    'vpn_vimeo_a': 9,
    'vpn_vimeo_b': 9,
    'vpn_youtube_a': 9,
    # VPN: Torrent
    'vpn_bittorrent': 10,
    # VPN VoIP
    'vpn_facebook_audio2': 11,
    'vpn_hangouts_audio1': 11,
    'vpn_hangouts_audio2': 11,
    'vpn_skype_audio1': 11,
    'vpn_skype_audio2': 11,
}

ID_TO_TRAFFIC = {
    0: 'Chat',
    1: 'Email',
    2: 'File Transfer',
    3: 'Streaming',
    4: 'Torrent',
    5: 'Voip',
    6: 'VPN: Chat',
    7: 'VPN: File Transfer',
    8: 'VPN: Email',
    9: 'VPN: Streaming',
    10: 'VPN: Torrent',
    11: 'VPN: Voip',
}


def read_pcap(path: Path):
    packets = PcapReader(str(path))

    return packets


def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS segment
    if DNS in packet:
        return True

    return False
