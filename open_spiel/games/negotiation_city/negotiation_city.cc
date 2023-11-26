// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/negotiation_city/negotiation_city.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>

#include "open_spiel/abseil-cpp/absl/random/poisson_distribution.h" // 引入泊松分布
#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h" // 引入均匀整数分布
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h" // 引入字符串连接功能
#include "open_spiel/abseil-cpp/absl/strings/str_join.h" // 引入字符串连接功能
#include "open_spiel/abseil-cpp/absl/strings/str_split.h" // 引入字符串分割功能
#include "open_spiel/spiel.h" // 引入OpenSpiel框架
#include "open_spiel/spiel_utils.h" // 引入OpenSpiel实用工具

// 包含 OpenSpiel 框架的一些核心库文件，用于处理字符串、概率分布和游戏逻辑等。
namespace open_spiel {
namespace negotiation_city {

// 声明命名空间 open_spiel 和子命名空间 negotiation_city，代码的主体部分将在这些命名空间内定义。

namespace {

// 匿名命名空间，用于内部实现，限制其作用域仅在当前文件中。

// Facts about the game  
// 定义游戏类型的基本属性，如游戏名称、动态类型、玩家数量等。此外，还定义了游戏的参数。

const GameType kGameType{
    /*short_name=*/"negotiation_city",
    /*long_name=*/"Negotiation_city",
    GameType::Dynamics::kSequential,   // 顺序动态
    GameType::ChanceMode::kSampledStochastic,  // 抽样随机
    GameType::Information::kImperfectInformation,  // 非完全信息
    GameType::Utility::kGeneralSum,   // 总和效用
    GameType::RewardModel::kTerminal,  // 终端奖励
    /*max_num_players=*/10, // Jin: added the player numbers
    /*min_num_players=*/10,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"enable_proposals", GameParameter(kDefaultEnableProposals)},
     {"enable_utterances", GameParameter(kDefaultEnableUtterances)},
     {"num_items", GameParameter(kDefaultNumItems)},
     {"num_symbols", GameParameter(kDefaultNumSymbols)},
     {"rng_seed", GameParameter(kDefaultSeed)},
     {"utterance_dim", GameParameter(kDefaultUtteranceDim)}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new NegotiationCityGame(params));
}

// Factory 函数用于根据给定的参数创建一个新的游戏实例。
REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// 将给定玩家的动作转换为字符串表示。
std::string TurnTypeToString(TurnType turn_type) {
  if (turn_type == TurnType::kProposal) {
    return "Proposal";
  } else if (turn_type == TurnType::kUtterance) {
    return "Utterance";  // Utterance: 发言
  } else {
    SpielFatalError("Unrecognized turn type");  // 未识别的回合类型错误
  }
}
}  // namespace

std::string NegotiationCityState::ActionToString(Player player,
                                             Action move_id) const {
  if (player == kChancePlayerId) {  // 若为概率玩家，则返回概率结果及其标识。
    return absl::StrCat("chance outcome ", move_id);  // 概率节点的动作
  } else {
    std::string action_string = "";
    if (turn_type_ == TurnType::kProposal) {
      if (move_id == parent_game_.NumDistinctProposals() - 1) {
        // 若提案编号对应协议达成，则标记为达成协议。
        absl::StrAppend(&action_string, "Proposal: Agreement reached!");  
      } else {
         // 否则，解码提案动作。
        std::vector<int> proposal = DecodeProposal(move_id);  // 解码提案
        std::string prop_str = absl::StrJoin(proposal, ", ");
        absl::StrAppend(&action_string, "Proposal: [", prop_str, "]");
      }
    } else {
         // 如果没有提案，则解码发言动作。
      std::vector<int> utterance = DecodeUtterance(move_id);  // 解码话语
      std::string utt_str = absl::StrJoin(utterance, ", ");
      absl::StrAppend(&action_string, ", Utterance: [", utt_str, "]");
    }
    return action_string;
  }
}

// IsTerminal 方法：判断当前游戏状态是否终止。
bool NegotiationCityState::IsTerminal() const {
  // If utterances are enabled, force the agent to utter something even when
  // they accept the proposal or run out of steps (i.e. on ther last turn).
  // 如果启用了话语，则即使在接受提案或步数耗尽时，也强制代理发表话语（即在最后一回合）。

  bool utterance_check =
      // 检查是否所有提案都有对应的发言。
      (enable_utterances_ ? utterances_.size() == proposals_.size() : true);
       // 若达成协议或提案数量达到最大步数，则游戏终止。
  return (agreement_reached_ || proposals_.size() >= max_steps_) &&
         utterance_check;
}

// Returns 方法：计算当前状态下的回报。
std::vector<double> NegotiationCityState::Returns() const {
  // added for multi players
  std::vector<double> returns(num_players_, 0.0);

  if (!IsTerminal() || !agreement_reached_) {
    // 若游戏未终止或未达成协议，则所有玩家回报为0。
    return std::vector<double>(num_players_, 0.0);
  }

  // 计算提出最终提案的玩家。
  int proposing_player = proposals_.size() % 2 == 1 ? 0 : 1;
  // 计算另一位玩家。
  int other_player = 1 - proposing_player;
  // 获取最终提案。
  const std::vector<int>& final_proposal = proposals_.back();

  // 初始化回报向量。
  std::vector<double> returns(num_players_, 0.0);
  for (int j = 0; j < num_items_; ++j) {
    returns[proposing_player] +=
        // 计算提案玩家的回报。
        agent_utils_[proposing_player][j] * final_proposal[j];
        // 计算另一玩家的回报。
        //  TODO 这里可能是限制2个以上玩家的问题所在
    returns[other_player] +=
        agent_utils_[other_player][j] * (item_pool_[j] - final_proposal[j]);
  }
// 返回计算后的回报。
  return returns;
}

// ObservationString 方法：为特定玩家生成当前游戏状态的观察字符串。
std::string NegotiationCityState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);  // 确保玩家ID大于等于0。
  SPIEL_CHECK_LT(player, num_players_);  // 确保玩家ID小于总玩家数。

  // 如果当前节点是概率节点，返回特定的无观察结果字符串。
  if (IsChanceNode()) {
    return "ChanceNode -- no observation";  // 对于概率节点，没有观察结果。
  }

  // 初始化观察结果字符串。
  std::string str = absl::StrCat("Max steps: ", max_steps_, "\n");  // 添加最大步数信息。
  absl::StrAppend(&str, "Item pool: ", absl::StrJoin(item_pool_, " "), "\n");  // 添加物品池信息。

  // 如果代理的效用向量不为空，则添加该玩家的效用信息。
  if (!agent_utils_.empty()) {
    absl::StrAppend(&str, "Agent ", player,
                    " util vec: ", absl::StrJoin(agent_utils_[player], " "),
                    "\n");  // 添加当前玩家的效用向量。
  }

  // 添加当前玩家信息和当前回合类型。
  absl::StrAppend(&str, "Current player: ", CurrentPlayer(), "\n");  // 当前玩家。
  absl::StrAppend(&str, "Turn Type: ", TurnTypeToString(turn_type_), "\n");  // 当前回合类型。

  // 如果存在提案记录，添加最近的提案信息。
  if (!proposals_.empty()) {
    absl::StrAppend(&str, "Most recent proposal: [",
                    absl::StrJoin(proposals_.back(), ", "), "]\n");  // 最新提案。
  }

  // 如果存在发言记录，添加最近的发言信息。
  if (!utterances_.empty()) {
    absl::StrAppend(&str, "Most recent utterance: [",
                    absl::StrJoin(utterances_.back(), ", "), "]\n");  // 最新发言。
  }

  // 返回构建好的观察结果字符串。
  return str;
}
// 1D vector with shape:
//   - Current player: kNumPlayers bits
//   - Current turn type: 2 bits
//   - Terminal status: 2 bits: (Terminal? and Agreement reached?)
//   - Context:
//     - item pool      (num_items * (max_quantity + 1) bits)
//     - my utilities   (num_items * (max_value + 1) bits)
//   - Last proposal:   (num_items * (max_quantity + 1) bits)
// If utterances are enabled, another:
//   - Last utterance:  (utterance_dim * num_symbols) bits)

// ObservationTensorShape 方法：返回表示游戏状态的观察张量的形状。
std::vector<int> NegotiationCityGame::ObservationTensorShape() const {
  // 返回一个包含各维度大小的向量，用于定义观察张量的形状。
  // 包括：当前玩家数、当前回合类型、游戏是否终止、游戏是否达成协议、物品池的大小、玩家的效用值、最后一个提案的大小。
  // 如果启用了发言（utterances），还包括发言的维度。
  return {kNumPlayers + 2 + 2 + (num_items_ * (kMaxQuantity + 1)) +
          (num_items_ * (kMaxValue + 1)) + (num_items_ * (kMaxQuantity + 1)) +
          (enable_utterances_ ? utterance_dim_ * num_symbols_ : 0)};
}

// ObservationTensor 方法：填充表示玩家观察的张量。
void NegotiationCityState::ObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);  // 检查玩家ID是否有效（非负）。
  SPIEL_CHECK_LT(player, num_players_);  // 检查玩家ID是否小于总玩家数。

  // 确保提供的张量尺寸与游戏预期的观察张量尺寸一致。
  SPIEL_CHECK_EQ(values.size(), parent_game_.ObservationTensorSize());
  // 初始化张量的所有值为0。
  std::fill(values.begin(), values.end(), 0);

  // 对于概率节点，不进行任何观察处理。
  if (IsChanceNode()) {
    return;  // 概率节点无观察。
  }


  // 1D vector with shape:
  //   - Current player: 2 bits
  //   - Turn type: 2 bits
  //   - Terminal status: 2 bits: (Terminal? and Agreement reached?)
  //   - Context:
  //     - item pool      (num_items * (max_quantity + 1) bits)
  //     - my utilities   (num_items * (max_value + 1) bits)
  //   - Last proposal:   (num_items * (max_quantity + 1) bits)
  // If utterances are enabled, another:
  //   - Last utterance:  (utterance_dim * num_symbols) bits)

  // 观察张量的构造过程，用于表示游戏的当前状态。
  // 观察张量包含以下部分：
  //   - 当前玩家：2位
  //   - 当前回合类型：2位
  //   - 游戏终止状态：2位（是否终止和是否达成协议）
  //   - 上下文信息：
  //     - 物品池：每个物品 (max_quantity + 1) 位
  //     - 玩家的效用值：每个物品 (max_value + 1) 位
  //   - 最后的提案：每个物品 (max_quantity + 1) 位
  //   - 如果启用了话语，则包括最后的话语：(utterance_dim * num_symbols) 位

  int offset = 0;  // 定义偏移量。

  // 设置当前玩家的位。
  if (!IsTerminal()) {
    values[offset + CurrentPlayer()] = 1;
  }
  offset += kNumPlayers;  // 更新偏移量。

  // 设置当前回合类型的位。
  if (turn_type_ == TurnType::kProposal) {
    values[offset] = 1;
  } else {
    values[offset + 1] = 1;
  }
  offset += 2;  // 更新偏移量。

  // 设置游戏终止状态的位。
  values[offset] = IsTerminal() ? 1 : 0;
  values[offset + 1] = agreement_reached_ ? 1 : 0;
  offset += 2;  // 更新偏移量。

  // 设置物品池的位。
  for (int item = 0; item < num_items_; ++item) {
    values[offset + item_pool_[item]] = 1;
    offset += kMaxQuantity + 1;  // 更新偏移量。
  }

  // 设置玩家的效用值的位。
  for (int item = 0; item < num_items_; ++item) {
    values[offset + agent_utils_[player][item]] = 1;
    offset += kMaxValue + 1;  // 更新偏移量。
  }

  // 设置最后的提案的位。
  if (!proposals_.empty()) {
    for (int item = 0; item < num_items_; ++item) {
      values[offset + proposals_.back()[item]] = 1;
      offset += kMaxQuantity + 1;  // 更新偏移量。
    }
  } else {
    offset += num_items_ * (kMaxQuantity + 1);  // 如果没有提案，更新偏移量。
  }

  // 设置最后的话语的位（如果启用了话语）。
  if (enable_utterances_) {
    if (!utterances_.empty()) {
      for (int dim = 0; dim < utterance_dim_; ++dim) {
        values[offset + utterances_.back()[dim]] = 1;
        offset += num_symbols_;  // 更新偏移量。
      }
    } else {
      offset += utterance_dim_ * num_symbols_;  // 如果没有话语，更新偏移量。
    }
  }

  // 检查偏移量是否与预期的张量大小一致。
  SPIEL_CHECK_EQ(offset, values.size());
}

// NegotiationCityState 类的构造函数
// 初始化 NegotiationCityState 类的实例，包括游戏状态和与游戏逻辑相关的变量。
NegotiationCityState::NegotiationCityState(std::shared_ptr<const Game> game)
    : State(game),  // 调用基类 State 的构造函数
      parent_game_(static_cast<const NegotiationCityGame&>(*game)),  // 强制类型转换，获取当前游戏的引用
      enable_proposals_(parent_game_.EnableProposals()),  // 初始化是否启用提案的标志
      enable_utterances_(parent_game_.EnableUtterances()),  // 初始化是否启用发言的标志
      num_items_(parent_game_.NumItems()),  // 初始化物品数量
      num_symbols_(parent_game_.NumSymbols()),  // 初始化符号数量
      utterance_dim_(parent_game_.UtteranceDim()),  // 初始化话语维度
      num_steps_(0),  // 初始化步数为0
      max_steps_(-1),  // 初始化最大步数为-1
      agreement_reached_(false),  // 初始化协议达成标志为假
      cur_player_(kChancePlayerId),  // 初始化当前玩家为概率玩家ID
      turn_type_(TurnType::kProposal),  // 初始化回合类型为提案
      item_pool_({}),  // 初始化物品池为空
      agent_utils_({}),  // 初始化玩家效用为空
      proposals_({}),  // 初始化提案列表为空
      utterances_({}) {}  // 初始化发言列表为空

// 返回当前玩家的ID
int NegotiationCityState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;  // 如果游戏结束，返回终端玩家ID，否则返回当前玩家ID
}

// From Sec 2.1 of the paper: "At each round (i) an item pool is sampled
// uniformly, instantiating a quantity (between 0 and 5) for each of the types
// and represented as a vector i \in {0...5}^3 and (ii) each agent j receives a
// utility function sampled uniformly, which specifies how rewarding one unit of
// each item is (with item rewards between 0 and 10, and with the constraint
// that there is at least one item with non-zero utility), represented as a
// vector u_j \in {0...10}^3".

// 确定物品池和玩家效用的方法
void NegotiationCityState::DetermineItemPoolAndUtilities() {
  // Generate max number of rounds (max number of steps for the episode): we
  // sample N between 4 and 10 at the start of each episode, according to a
  // truncated Poissondistribution with mean 7, as done in the Cao et al. '18
  // paper.
  // 根据泊松分布生成一个游戏回合的最大数量，平均值为7，范围在4到10之间
  max_steps_ = -1;
  absl::poisson_distribution<int> steps_dist(7.0);
  while (!(max_steps_ >= 4 && max_steps_ <= 10)) {
    max_steps_ = steps_dist(*parent_game_.RNG());
  }

  // Generate the pool of items.
  // 生成物品池，每个物品的数量随机分布在0到kMaxQuantity之间
  absl::uniform_int_distribution<int> quantity_dist(0, kMaxQuantity);
  for (int i = 0; i < num_items_; ++i) {
    item_pool_.push_back(quantity_dist(*parent_game_.RNG()));
  }

  // Generate agent utilities.
  // 为每个玩家生成效用向量，每个物品的效用值在0到kMaxValue之间
  absl::uniform_int_distribution<int> util_dist(0, kMaxValue);
  for (int i = 0; i < num_players_; ++i) {
    agent_utils_.push_back({});
    int sum_util = 0;
    // 确保至少有一个物品的效用值非零
    while (sum_util == 0) {
      for (int j = 0; j < num_items_; ++j) {
        agent_utils_[i].push_back(util_dist(*parent_game_.RNG()));
        sum_util += agent_utils_[i].back();
      }
    }
  }
}

// 初始化游戏回合
void NegotiationCityState::InitializeEpisode() {
  cur_player_ = 0;  // 设置当前玩家为第一个玩家
  turn_type_ = TurnType::kProposal;  // 设置回合类型为提案
}

// 应用动作的方法
void NegotiationCityState::DoApplyAction(Action move_id) {
  if (IsChanceNode()) {
    DetermineItemPoolAndUtilities();  // 如果是概率节点，确定物品池和玩家效用
    InitializeEpisode();  // 初始化游戏回合
  } else {
    if (turn_type_ == TurnType::kProposal) {
      if (move_id == parent_game_.NumDistinctProposals() - 1) {
        agreement_reached_ = true;  // 如果动作ID表示达成协议，则设置协议达成标志为真
      } else {
        std::vector<int> proposal = DecodeProposal(move_id);  // 否则，解码提案
        proposals_.push_back(proposal);  // 添加提案到提案列表
      }

      if (enable_utterances_) {
        turn_type_ = TurnType::kUtterance;  // 如果启用了话语，设置回合类型为发言
      } else {
        // cur_player_ = 1 - cur_player_;  // 否则，切换到下一个玩家
        cur_player_ = (cur_player_ + 1) % num_players_;  // Added: Move to the next player.
        // cur_player_ = 3;  // Added: Move to the next player.

        // Output the current player.
        std::cout << "Current player after proposal: " << cur_player_ << std::endl;
      }
    } else {
      SPIEL_CHECK_TRUE(enable_utterances_);
      std::vector<int> utterance = DecodeUtterance(move_id);  // 解码发言
      utterances_.push_back(utterance);  // 添加发言到发言列表
      turn_type_ = TurnType::kProposal;  // 设置回合类型为提案
      // cur_player_ = 1 - cur_player_;  // 切换到下一个玩家
      cur_player_ = (cur_player_ + 1) % num_players_;  // Added: Move to the next player.
      // cur_player_ = 4;
      // Output the current player.
      std::cout << "Current player after utterance: " << cur_player_ << std::endl;
    }
  }
}


// NextProposal 方法：生成下一个提案。
bool NegotiationCityState::NextProposal(std::vector<int>* proposal) const {
  // Starting from the right, move left trying to increase the value. When
  // successful, increment the value and set all the right digits back to 0.
  // 从右侧开始，向左移动，尝试增加值。成功后，增加该值，并将所有右侧的数字重置为0。
  // 此方法用于遍历所有可能的提案组合。

  // 从物品列表的最后一个物品开始，向前遍历每个物品。
  for (int i = num_items_ - 1; i >= 0; --i) {
    // 检查当前物品的数量是否可以增加（即是否小于或等于该物品在物品池中的数量）。
    if ((*proposal)[i] + 1 <= item_pool_[i]) {
      // 如果可以增加，将该物品的数量加1。
      (*proposal)[i]++;

      // 将当前物品后面所有物品的数量重置为0。
      // 这是为了确保每次只改变一个物品的数量，而其它物品数量重置，以便遍历所有可能的提案组合。
      for (int j = i + 1; j < num_items_; ++j) {
        (*proposal)[j] = 0;
      }

      // 返回 true，表示成功生成了下一个提案。
      return true;
    }
  }

  // 如果所有物品的数量都不能增加，则返回 false，表示没有更多的提案可以生成。
  return false;
}

// DecodeInteger 方法：将一个编码后的整数值解码为一个整数向量。
std::vector<int> NegotiationCityState::DecodeInteger(int encoded_value, int dimensions, int num_digit_values) const {
  std::vector<int> decoded(dimensions, 0);  // 初始化一个大小为 dimensions，值全部为0的向量。
  int i = dimensions - 1;
  while (encoded_value > 0) {
    SPIEL_CHECK_GE(i, 0);  // 检查索引 i 是否有效。
    SPIEL_CHECK_LT(i, dimensions);
    decoded[i] = encoded_value % num_digit_values;  // 计算当前位的值。
    encoded_value /= num_digit_values;  // 将编码值除以基数，准备计算下一位。
    i--;  // 移动到下一个更高位。
  }
  return decoded;
}

// EncodeInteger 方法：将一个整数向量编码为一个整数值。
int NegotiationCityState::EncodeInteger(const std::vector<int>& container, int num_digit_values) const {
  int encoded_value = 0;
  for (int digit : container) {
    encoded_value = encoded_value * num_digit_values + digit;  // 将每个数字乘以基数并加到编码值上。
  }
  return encoded_value;
}

// EncodeProposal 方法：将提案向量编码为一个整数。
Action NegotiationCityState::EncodeProposal(const std::vector<int>& proposal) const {
  SPIEL_CHECK_EQ(proposal.size(), num_items_);  // 检查提案大小是否正确。
  return EncodeInteger(proposal, kMaxQuantity + 1);  // 调用 EncodeInteger 进行编码。
}

// EncodeUtterance 方法：将话语向量编码为一个整数。
Action NegotiationCityState::EncodeUtterance(const std::vector<int>& utterance) const {
  SPIEL_CHECK_EQ(utterance.size(), utterance_dim_);  // 检查话语大小是否正确。
  // 话语ID从 NumDistinctProposals() 开始，因此在编码时要进行偏移。
  return parent_game_.NumDistinctProposals() + EncodeInteger(utterance, num_symbols_);
}

// DecodeProposal 方法：将一个编码后的提案解码为提案向量。
std::vector<int> NegotiationCityState::DecodeProposal(int encoded_proposal) const {
  return DecodeInteger(encoded_proposal, num_items_, kMaxQuantity + 1);  // 调用 DecodeInteger 进行解码。
}


// DecodeUtterance 方法：将一个编码后的话语解码为话语向量。
std::vector<int> NegotiationCityState::DecodeUtterance(int encoded_utterance) const {
  // 话语ID从 NumDistinctProposals() 开始，因此在解码时要进行偏移。
  return DecodeInteger(encoded_utterance - parent_game_.NumDistinctProposals(), utterance_dim_, num_symbols_);
}

// LegalActions 方法：返回当前状态下的合法动作列表。
std::vector<Action> NegotiationCityState::LegalActions() const {
  if (IsChanceNode()) {
    // 如果是概率节点，返回概率节点的合法动作。
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    // 如果游戏已结束，没有合法动作。
    return {};
  } else if (turn_type_ == TurnType::kProposal) {
    // 如果当前回合类型为提案，则构建提案动作列表。
    std::vector<Action> legal_actions;

    // 构建初始提案并加入到动作列表。
    std::vector<int> proposal(num_items_, 0);
    legal_actions.push_back(EncodeProposal(proposal));

    // 遍历所有可能的提案，并将它们加入到动作列表。
    while (NextProposal(&proposal)) {
      legal_actions.push_back(EncodeProposal(proposal));
    }

    // 如果已有提案，则添加表示达成协议的动作。
    if (!proposals_.empty()) {
      legal_actions.push_back(parent_game_.NumDistinctProposals() - 1);
    }

    return legal_actions;
  } else {
    // 如果启用了话语，则返回话语的合法动作。
    SPIEL_CHECK_TRUE(enable_utterances_);
    SPIEL_CHECK_FALSE(parent_game_.LegalUtterances().empty());
    return parent_game_.LegalUtterances();
  }
}

// ChanceOutcomes 方法：返回概率节点的可能结果。
std::vector<std::pair<Action, double>> NegotiationCityState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  // 因为游戏是随机采样的，所以只有一个结果，且所有随机性都在 ApplyAction 中处理。
  // 返回一个包含单个结果的向量，其概率为1.0。
  std::vector<std::pair<Action, double>> outcomes = {std::make_pair(0, 1.0)};
  return outcomes;
}

// ToString 方法：返回当前游戏状态的字符串表示。
std::string NegotiationCityState::ToString() const {
  if (IsChanceNode()) {
    // 如果是概率节点，返回概率节点的字符串表示。
    return "Initial chance node";
  }

  // 构建游戏状态的字符串表示。
  std::string str = absl::StrCat("Max steps: ", max_steps_, "\n");
  absl::StrAppend(&str, "Item pool: ", absl::StrJoin(item_pool_, " "), "\n");

  // 添加每位玩家的效用向量信息。// TODO 这里只有两个 0和1
  if (!agent_utils_.empty()) {
    for (int i = 0; i < num_players_; ++i) {
      absl::StrAppend(&str, "Agent ", i, " util vec: ", absl::StrJoin(agent_utils_[i], " "), "\n");
    }
  }

  // 添加当前玩家和回合类型信息。
  absl::StrAppend(&str, "Current player001: ", cur_player_, "\n");
  absl::StrAppend(&str, "Turn Type: ", TurnTypeToString(turn_type_), "\n");

  // 添加每个玩家的提案和发言信息。
  for (int i = 0; i < proposals_.size(); ++i) {
    // absl::StrAppend(&str, "Player ", i % 2, " proposes: [", absl::StrJoin(proposals_[i], ", "), "]");
    absl::StrAppend(&str, "Player ", i % num_players_, " proposes: [", absl::StrJoin(proposals_[i], ", "), "]");

    if (enable_utterances_ && i < utterances_.size()) {
      absl::StrAppend(&str, " utters: [", absl::StrJoin(utterances_[i], ", "), "]");
    }
    absl::StrAppend(&str, "\n");
  }

  // 如果达成了协议，添加相应的信息。
  if (agreement_reached_) {
    absl::StrAppend(&str, "Agreement reached!\n");
  }

  return str;
}

// NegotiationCityState 类的 Clone 方法
// 用于创建当前游戏状态的一个新副本。
std::unique_ptr<State> NegotiationCityState::Clone() const {
  // 返回一个指向新创建的 NegotiationCityState 副本的智能指针。
  return std::unique_ptr<State>(new NegotiationCityState(*this));
}

// NegotiationCityGame 类的构造函数
NegotiationCityGame::NegotiationCityGame(const GameParameters& params)
    : Game(kGameType, params),  // 调用基类 Game 的构造函数。
      // 从参数中读取并初始化游戏的配置。
      enable_proposals_(ParameterValue<bool>("enable_proposals", kDefaultEnableProposals)),
      enable_utterances_(ParameterValue<bool>("enable_utterances", kDefaultEnableUtterances)),
      num_items_(ParameterValue<int>("num_items", kDefaultNumItems)),
      num_symbols_(ParameterValue<int>("num_symbols", kDefaultNumSymbols)),
      utterance_dim_(ParameterValue<int>("utterance_dim", kDefaultUtteranceDim)),
      seed_(ParameterValue<int>("rng_seed", kDefaultSeed)),
      legal_utterances_({}),
      rng_(new std::mt19937(seed_ >= 0 ? seed_ : std::mt19937::default_seed)) {
  // 构造合法的话语动作列表。
  ConstructLegalUtterances();
}

// 构造合法话语动作的方法。
void NegotiationCityGame::ConstructLegalUtterances() {
  if (enable_utterances_) {
    legal_utterances_.resize(NumDistinctUtterances());
    for (int i = 0; i < NumDistinctUtterances(); ++i) {
      legal_utterances_[i] = NumDistinctProposals() + i;
    }
  }
}

// 返回游戏的最大长度。
int NegotiationCityGame::MaxGameLength() const {
  if (enable_utterances_) {
    // 如果启用了话语，则每一步包含两个回合：提案和发言。
    return 2 * kMaxSteps;  // Every step is two turns: proposal, then utterance.
  } else {
    // 如果没有启用话语，返回最大步数。
    return kMaxSteps;
  }
}

// 返回不同话语的数量。
int NegotiationCityGame::NumDistinctUtterances() const {
  // 根据话语的维度和符号数量计算不同话语的总数。
  return static_cast<int>(std::pow(num_symbols_, utterance_dim_));
}

// 返回不同提案的数量。
int NegotiationCityGame::NumDistinctProposals() const {
  // 每个物品的数量可以是 {0, 1, ..., MaxQuantity}，加上一个特殊的“达成协议”动作。
  // Every slot can hold { 0, 1, ..., MaxQuantity }, and there is an extra
  // one at the end for the special "agreement reached" action.
  return static_cast<int>(std::pow(kMaxQuantity + 1, num_items_)) + 1;
}

// See the header for an explanation of how these are encoded.
int NegotiationCityGame::NumDistinctActions() const {
  if (enable_utterances_) {
    // 如果启用了话语，动作包括提案和话语。
    return NumDistinctProposals() + NumDistinctUtterances();
  } else {
    // Proposals are always possible.
    // 如果没有启用话语，只有提案动作。
    return NumDistinctProposals();
  }
}

// Serialize 方法：序列化当前游戏状态为字符串。
std::string NegotiationCityState::Serialize() const {
  if (IsChanceNode()) {
    // 对于概率节点，返回特定字符串。
    return "chance";
  } else {
    // 构建表示当前游戏状态的字符串。
    std::string state_str = "";
    absl::StrAppend(&state_str, MaxSteps(), "\n");
    absl::StrAppend(&state_str, absl::StrJoin(ItemPool(), " "), "\n");
    for (int p = 0; p < NumPlayers(); ++p) {
      absl::StrAppend(&state_str, absl::StrJoin(AgentUtils()[p], " "), "\n");
    }
    absl::StrAppend(&state_str, HistoryString(), "\n");
    return state_str;
  }
}

// NegotiationCityGame 类的 DeserializeState 方法
// 用于根据提供的字符串反序列化游戏状态。
std::unique_ptr<State> NegotiationCityGame::DeserializeState(const std::string& str) const {
  if (str == "chance") {
    // 如果字符串表示概率节点，则创建并返回初始状态。
    return NewInitialState();
  } else {
    // 将字符串分割为多行，每行代表游戏状态的一部分。
    std::vector<std::string> lines = absl::StrSplit(str, '\n');
    std::unique_ptr<State> state = NewInitialState();  // 创建新的初始状态。
    SPIEL_CHECK_EQ(lines.size(), 5);  // 确保字符串行数符合预期。
    NegotiationCityState& nstate = static_cast<NegotiationCityState&>(*state);
    // Take the chance action, but then reset the quantities. Make sure game's
    // RNG state is not advanced during deserialization so copy it beforehand
    // in order to be able to restore after the chance action.
    // 复制 RNG 状态以在反序列化过程中保持不变。
    std::unique_ptr<std::mt19937> rng = std::make_unique<std::mt19937>(*rng_);
    nstate.ApplyAction(0);  // 应用概率动作。
    rng_ = std::move(rng);  // 恢复 RNG 状态。

    // 清空物品池和玩家效用。
    nstate.ItemPool().clear();
    nstate.AgentUtils().clear();

    // Max steps
    // 设置最大步数。
    nstate.SetMaxSteps(std::stoi(lines[0]));

    // Item pool.
    // 从字符串中恢复物品池。
    std::vector<std::string> parts = absl::StrSplit(lines[1], ' ');
    for (const auto& part : parts) {
      nstate.ItemPool().push_back(std::stoi(part));
    }

    // Agent utilities.
    // 从字符串中恢复玩家效用。
    for (Player player : {0, 1}) {
      parts = absl::StrSplit(lines[2 + player], ' ');
      nstate.AgentUtils().push_back({});
      for (const auto& part : parts) {
        nstate.AgentUtils()[player].push_back(std::stoi(part));
      }
    }

    // 设置当前玩家。
    nstate.SetCurrentPlayer(0);

    // Actions.
    // 应用历史动作。
    if (lines.size() == 5) {
      parts = absl::StrSplit(lines[4], ' ');
      // Skip the first one since it is the chance node.
      for (int i = 1; i < parts.size(); ++i) {
        Action action = static_cast<Action>(std::stoi(parts[i]));
        nstate.ApplyAction(action);
      }
    }
    return state;
  }
}

// GetRNGState 方法：获取当前随机数生成器的状态。
std::string NegotiationCityGame::GetRNGState() const {
  std::ostringstream rng_stream;
  rng_stream << *rng_;  // 将 RNG 状态写入字符串流。
  return rng_stream.str();  // 返回 RNG 状态字符串。
}

// SetRNGState 方法：设置随机数生成器的状态。
void NegotiationCityGame::SetRNGState(const std::string& rng_state) const {
  if (rng_state.empty()) return;  // 如果状态字符串为空，不进行操作。
  std::istringstream rng_stream(rng_state);
  rng_stream >> *rng_;  // 从字符串流中读取并设置 RNG 状态。
}

}  // namespace negotiation city
}  // namespace open_spiel
