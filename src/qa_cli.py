"""CLI entry point for the Joplin QA system."""
import argparse
import json
import time

import pathmagic

with pathmagic.Context():
    from func.logme import log
    from src.qa_config import CONFIG
    from src.qa_system import QASystem

__all__ = ["parse_args", "interactive_mode", "main"]


def parse_args():
    parser = argparse.ArgumentParser(description="Joplin笔记智能问答系统")
    parser.add_argument("--question", "-q", type=str, help="直接提问的问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互模式")
    parser.add_argument(
        "--model", type=str, default=CONFIG["qa_ollama_chat_model"],
        help=f"聊天模型名称（默认：{CONFIG['qa_ollama_chat_model']}）",
    )
    parser.add_argument(
        "--use-cloud", action="store_true", default=False,
        help=f"使用云端模型（默认：{CONFIG.get('cloud_model', 'deepseek-chat')}）",
    )
    parser.add_argument(
        "--max-notes", type=int, default=CONFIG["max_retrieved_notes"],
        help=f"最大检索笔记数（默认：{CONFIG['max_retrieved_notes']}）",
    )
    parser.add_argument("--clear-history", action="store_true", help="清空对话历史")
    parser.add_argument("--stats", action="store_true", help="显示系统统计信息")
    return parser.parse_args()


def interactive_mode(qa_system: QASystem):
    print("\n" + "=" * 60)
    print("Joplin笔记智能问答系统 - 交互模式")
    print("=" * 60)
    print("命令说明:")
    print("  /quit 或 /exit - 退出")
    print("  /clear - 清空对话历史")
    print("  /stats - 显示统计信息")
    print("  /help - 显示帮助")
    print("=" * 60)

    while True:
        try:
            question = input("\n问: ").strip()
            if not question:
                continue

            if question.lower() in ["/quit", "/exit"]:
                print("再见！")
                break
            elif question.lower() == "/clear":
                qa_system.clear_history()
                print("对话历史已清空")
                continue
            elif question.lower() == "/stats":
                stats = qa_system.get_statistics()
                print(f"\n系统统计:")
                print(f"  数据库笔记数: {stats['total_notes_in_db']}")
                print(f"  对话历史数: {stats['conversation_history_count']}")
                print(f"  嵌入模型: {stats['config']['ollama_embedding_model']}")
                print(f"  聊天模型: {stats['config']['qa_ollama_chat_model']}")
                print(f"  使用云端模型: {stats['config']['using_cloud']}")
                continue
            elif question.lower() == "/help":
                print("\n可用命令:")
                print("  /quit, /exit - 退出程序")
                print("  /clear - 清空对话历史")
                print("  /stats - 显示系统统计")
                print("  /help - 显示此帮助")
                continue

            print("\n思考中...", end="", flush=True)
            start_time = time.time()
            result = qa_system.ask(question)
            elapsed_time = time.time() - start_time

            print(f"\r答: {result['answer']}")

            num_for_show = 3
            if result["is_based_on_notes"] and result["relevant_notes"]:
                print(
                    f"\n来源 ({len(result['relevant_notes'])}块相关文本块)，"
                    f"仅显示前{num_for_show}:"
                )
                for i, note in enumerate(result["relevant_notes"][:num_for_show], 1):
                    similarity = note["similarity"]
                    tags = note["metadata"].get("tags", "无标签")
                    print(f"  {i}. 相似度: {similarity:.2f} | 标签: {tags}")
            if result["sources"]:
                print(
                    f"\n以上文本块来源 ({len(result['sources'])}条相关笔记)，"
                    f"仅显示前{num_for_show}:"
                )
                for i, note in enumerate(result["sources"][:num_for_show], 1):
                    print(f"  {i}. 《{note['title']}》")

            print(
                f"\n[处理时间: {elapsed_time:.2f}秒 | "
                f"上下文长度: {result.get('context_length', 0)}字符]"
            )

        except KeyboardInterrupt:
            print("\n\n退出交互模式")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    args = parse_args()

    from joplinai import CONFIG as joplinai_config
    dynamic_config = {**CONFIG.copy(), **joplinai_config}
    dynamic_config["qa_ollama_chat_model"] = args.model
    dynamic_config["cloud_model"] = args.use_cloud if args.use_cloud else dynamic_config.get("cloud_model", "deepseek-chat")
    dynamic_config["max_retrieved_notes"] = args.max_notes

    log.info("初始化Joplin问答系统...")
    qa_system = QASystem(dynamic_config)

    if not qa_system.vector_db.collection:
        log.error("向量数据库未找到或无法加载！")
        log.error("请先运行 joplinai.py 进行笔记向量化")
        return

    if args.clear_history:
        qa_system.clear_history()
        print("对话历史已清空")
        return

    if args.stats:
        stats = qa_system.get_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    if args.question:
        print(f"问题: {args.question}")
        print("\n思考中...")
        result = qa_system.ask(args.question)
        print(f"\n答案: {result['answer']}")
        if result["is_based_on_notes"] and result["relevant_notes"]:
            print(f"\n基于 {len(result['relevant_notes'])} 条相关文本块:")
            for note in result["relevant_notes"]:
                print(f"  - 相似度: {note['similarity']:.2f}")
        return

    if args.interactive or (
        not args.question and not args.stats and not args.clear_history
    ):
        interactive_mode(qa_system)
        return


if __name__ == "__main__":
    main()
