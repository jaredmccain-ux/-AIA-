import win32com.client as win32
from win32com.client import constants
import os

def delete_last_page_of_word(file_path):
    # 验证文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    try:
        # 1. 启动Word应用（不可见模式）
        word_app = win32.DispatchEx("Word.Application")
        word_app.Visible = False  # 后台运行，不弹出Word窗口
        word_app.DisplayAlerts = 0  # 抑制警告弹窗

        # 2. 打开目标文档（只读=False，可修改）
        doc = word_app.Documents.Open(os.path.abspath(file_path))

        # 3. 获取文档总页数
        total_pages = doc.ComputeStatistics(constants.wdStatisticPages)
        print(f"文档当前总页数：{total_pages}")

        if total_pages <= 1:
            print("警告：文档仅1页，删除后将为空文档！")
            confirm = input("是否继续？（y/n）：")
            if confirm.lower() != "y":
                doc.Close(SaveChanges=0)
                word_app.Quit()
                print("操作取消。")
                return

        # 4. 跳转到最后一页的起始位置，选中并删除最后一页内容
        selection = word_app.Selection
        # 移动到文档末尾前一个字符（避免选中分页符导致删除异常）
        selection.EndKey(Unit=constants.wdStory)  # 到文档末尾
        selection.MoveLeft(Unit=constants.wdCharacter, Count=1)  # 左移1字符
        # 选中从当前位置到文档开头的内容（反向选中最后一页）
        selection.ExtendToDocumentStart()
        # 反向选中最后一页：先到文档末尾，再向前选中1页
        selection.EndKey(Unit=constants.wdStory)
        selection.MoveUp(Unit=constants.wdPage, Count=1, Extend=1)  # 向上选中1页

        # 5. 删除选中的最后一页内容
        selection.Delete()
        print("最后一页已删除！")

        # 6. 保存文档并关闭
        doc.Save()
        doc.Close(SaveChanges=0)
        word_app.Quit()
        print(f"操作完成！文件已保存至：{file_path}")

    except Exception as e:
        print(f"错误：删除失败！原因：{str(e)}")
        # 异常时强制关闭Word进程，避免残留
        try:
            doc.Close(SaveChanges=0)
            word_app.Quit()
        except:
            pass

# ------------------- 调用函数 -------------------
if __name__ == "__main__":
    # 替换为你的Word文档路径（支持.doc和.docx）
    WORD_FILE_PATH = r"E:\你的文档路径\人工智能2304班-U202315168-李易阳-传感技术实验报告.doc"
    delete_last_page_of_word(WORD_FILE_PATH)