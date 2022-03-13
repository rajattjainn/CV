


def parse_cfg(cfgfile):
    net_cfg = open(cfgfile, "rt")
    cfg_lines = list(net_cfg)
    cfg_lines = [line.strip() for line in cfg_lines]
    cfg_lines = list(filter(None, cfg_lines))
    block = {}
    blocks = []

    for line in cfg_lines:
        
        if line.startswith("#"):
            continue
        elif line.startswith("["):
            if len(block) > 0:
                blocks.append(block)
                block = {}
            section = line[1:-1]
            block["type"] = section.lstrip().rstrip()
        
        else:
            key, value = line.split("=")
            block[key.rstrip().lstrip()] = value.lstrip().rstrip()
        
    blocks.append(block)
    return blocks
            


blocks = parse_cfg("cfg/yolov3.cfg")

for block in blocks:
    print (block)
    print ("\n\n")

# print (parse_cfg("cfg/yolov3.cfg"))
