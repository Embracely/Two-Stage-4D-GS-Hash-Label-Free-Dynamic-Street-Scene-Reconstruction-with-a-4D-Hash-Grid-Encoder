"""
S3Gaussian基础配置类
"""

class BaseConfig:
    """
    所有配置类的基类
    """
    def __init__(self):
        pass
    
    def extract(self, args):
        """
        从配置类中提取参数
        
        Args:
            args: 命令行参数
            
        Returns:
            包含配置参数的命名空间
        """
        import argparse
        
        # 创建命名空间
        namespace = argparse.Namespace()
        
        # 复制所有非私有属性
        for key, value in vars(self).items():
            if not key.startswith('_'):
                setattr(namespace, key, value)
        
        # 复制类属性
        for key in dir(self.__class__):
            if not key.startswith('_') and key != 'extract':
                value = getattr(self.__class__, key)
                if not callable(value):
                    setattr(namespace, key, value)
        
        return namespace 