#!/bin/bash

# 应用哈希编码器补丁到train.py文件
# 此脚本将修改train.py文件以支持哈希编码器

echo "开始应用哈希编码器补丁到train.py..."

# 备份原始文件
cp train.py train.py.backup
echo "已备份原始train.py文件到train.py.backup"

# 添加导入语句
echo "添加导入语句..."
if ! grep -q "from scene.deformation_hash import deform_network_hash" train.py; then
    sed -i '1,20s/from scene.deformation import deform_network/from scene.deformation import deform_network\nfrom scene.deformation_hash import deform_network_hash\nfrom scene.hash_encoder import HashEncoder4D/' train.py
    echo "成功添加导入语句"
else
    echo "导入语句已存在，跳过添加"
fi

# 添加命令行参数
echo "添加命令行参数..."
if ! grep -q "parser.add_argument(\"--encoder_type\"" train.py; then
    sed -i '/parser.add_argument("--help_training_modes", action="store_true", help="显示训练模式帮助信息")/a \    parser.add_argument("--encoder_type", type=str, default="hexplane", choices=["hexplane", "hash"], help="Type of encoder to use")' train.py
    echo "成功添加encoder_type参数"
else
    echo "encoder_type参数已存在，跳过添加"
fi

# 添加哈希配置初始化
echo "添加哈希配置初始化..."
if ! grep -q "args.encoder_type == \"hash\" and not hasattr(args, \"hash_config\")" train.py; then
    sed -i '/args = parser.parse_args(sys.argv\[1:\])/a \    if args.encoder_type == "hash" and not hasattr(args, "hash_config"):\n        # Set default hash encoder parameters if not specified in config\n        args.hash_config = {\n            "n_levels": 16,\n            "min_resolution": 16,\n            "max_resolution": 512,\n            "log2_hashmap_size": 15,\n            "feature_dim": 2,\n        }' train.py
    echo "成功添加哈希配置初始化"
else
    echo "哈希配置初始化已存在，跳过添加"
fi

# 修改GaussianModel初始化
echo "修改GaussianModel初始化..."
if ! grep -q "Using Hash Encoder for deformation network" train.py; then
    sed -i 's/self._deformation = deform_network(args)/# Initialize deformation network based on encoder_type\n        if hasattr(args, "encoder_type") and args.encoder_type == "hash":\n            print("Using Hash Encoder for deformation network")\n            self._deformation = deform_network_hash(args)\n        else:\n            print("Using HexPlane for deformation network")\n            self._deformation = deform_network(args)/' train.py
    echo "成功修改GaussianModel初始化"
else
    echo "GaussianModel初始化已修改，跳过修改"
fi

# 修改正则化函数
echo "修改正则化函数..."
if ! grep -q "Check if using hash encoder or hexplane" train.py; then
    # 找到_plane_regulation函数的位置
    PLANE_REG_LINE=$(grep -n "_plane_regulation" train.py | head -n 1 | cut -d':' -f1)
    if [ ! -z "$PLANE_REG_LINE" ]; then
        # 提取函数定义到下一个函数定义之间的内容
        NEXT_FUNC_LINE=$(tail -n +$((PLANE_REG_LINE+1)) train.py | grep -n "def " | head -n 1 | cut -d':' -f1)
        NEXT_FUNC_LINE=$((PLANE_REG_LINE + NEXT_FUNC_LINE))
        
        # 替换函数内容
        sed -i "${PLANE_REG_LINE},${NEXT_FUNC_LINE}s/def _plane_regulation(self):.*/def _plane_regulation(self):\n        # Check if using hash encoder or hexplane\n        if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'grids'):\n            # HexPlane regulation\n            multi_res_grids = self._deformation.deformation_net.grid.grids\n            total = 0\n            for grids in multi_res_grids:\n                if len(grids) == 3:\n                    time_grids = []\n                else:\n                    time_grids = [0, 1, 3]\n                for grid_id in time_grids:\n                    total += compute_plane_smoothness(grids[grid_id])\n            return total\n        else:\n            # Hash encoder regulation - apply L2 regularization on hash table parameters\n            total = 0\n            if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'hash_tables'):\n                hash_tables = self._deformation.deformation_net.grid.hash_tables\n                for table in hash_tables:\n                    total += torch.mean(torch.square(table.weight))\n            return total * 0.01  # Scale factor to match hexplane regulation magnitude/" train.py
        echo "成功修改_plane_regulation函数"
    fi

    # 找到_time_regulation函数的位置
    TIME_REG_LINE=$(grep -n "_time_regulation" train.py | head -n 1 | cut -d':' -f1)
    if [ ! -z "$TIME_REG_LINE" ]; then
        # 提取函数定义到下一个函数定义之间的内容
        NEXT_FUNC_LINE=$(tail -n +$((TIME_REG_LINE+1)) train.py | grep -n "def " | head -n 1 | cut -d':' -f1)
        NEXT_FUNC_LINE=$((TIME_REG_LINE + NEXT_FUNC_LINE))
        
        # 替换函数内容
        sed -i "${TIME_REG_LINE},${NEXT_FUNC_LINE}s/def _time_regulation(self):.*/def _time_regulation(self):\n        # Check if using hash encoder or hexplane\n        if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'grids'):\n            # HexPlane regulation\n            multi_res_grids = self._deformation.deformation_net.grid.grids\n            total = 0\n            for grids in multi_res_grids:\n                if len(grids) == 3:\n                    time_grids = []\n                else:\n                    time_grids = [2, 4, 5]\n                for grid_id in time_grids:\n                    total += compute_plane_smoothness(grids[grid_id])\n            return total\n        else:\n            # For hash encoder, apply temporal smoothness by L2 regularization on consecutive time steps\n            # This is a simplified approach since hash tables don't have explicit time planes\n            return self._plane_regulation() * 0.5  # Use a fraction of spatial regularization/" train.py
        echo "成功修改_time_regulation函数"
    fi

    # 找到_l1_regulation函数的位置
    L1_REG_LINE=$(grep -n "_l1_regulation" train.py | head -n 1 | cut -d':' -f1)
    if [ ! -z "$L1_REG_LINE" ]; then
        # 提取函数定义到下一个函数定义之间的内容
        NEXT_FUNC_LINE=$(tail -n +$((L1_REG_LINE+1)) train.py | grep -n "def " | head -n 1 | cut -d':' -f1)
        NEXT_FUNC_LINE=$((L1_REG_LINE + NEXT_FUNC_LINE))
        
        # 替换函数内容
        sed -i "${L1_REG_LINE},${NEXT_FUNC_LINE}s/def _l1_regulation(self):.*/def _l1_regulation(self):\n        # Check if using hash encoder or hexplane\n        if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'grids'):\n            # HexPlane regulation\n            multi_res_grids = self._deformation.deformation_net.grid.grids\n            total = 0.0\n            for grids in multi_res_grids:\n                if len(grids) == 3:\n                    continue\n                else:\n                    spatiotemporal_grids = [2, 4, 5]\n                for grid_id in spatiotemporal_grids:\n                    total += torch.abs(1 - grids[grid_id]).mean()\n            return total\n        else:\n            # For hash encoder, apply L1 regularization on hash table parameters\n            total = 0.0\n            if hasattr(self._deformation.deformation_net, 'grid') and hasattr(self._deformation.deformation_net.grid, 'hash_tables'):\n                hash_tables = self._deformation.deformation_net.grid.hash_tables\n                for table in hash_tables:\n                    total += torch.abs(table.weight).mean()\n            return total * 0.01  # Scale factor to match hexplane regulation magnitude/" train.py
        echo "成功修改_l1_regulation函数"
    fi
    echo "正则化函数修改完成"
else
    echo "正则化函数已修改，跳过修改"
fi

# 检查是否有重复的encoder_type参数定义
ENCODER_TYPE_COUNT=$(grep -c "parser.add_argument(\"--encoder_type\"" train.py)
if [ "$ENCODER_TYPE_COUNT" -gt 1 ]; then
    echo "警告：检测到重复的encoder_type参数定义，尝试修复..."
    # 找到第一个encoder_type参数定义的行号
    FIRST_LINE=$(grep -n "parser.add_argument(\"--encoder_type\"" train.py | head -n 1 | cut -d':' -f1)
    # 找到第二个encoder_type参数定义的行号
    SECOND_LINE=$(grep -n "parser.add_argument(\"--encoder_type\"" train.py | tail -n 1 | cut -d':' -f1)
    
    if [ "$FIRST_LINE" != "$SECOND_LINE" ]; then
        # 删除第二个encoder_type参数定义
        sed -i "${SECOND_LINE}d" train.py
        echo "成功删除重复的encoder_type参数定义"
    fi
fi

echo "哈希编码器补丁已应用完成！"
echo "如果需要恢复原始文件，请运行: mv train.py.backup train.py" 