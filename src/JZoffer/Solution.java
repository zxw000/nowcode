package JZoffer;
import java.util.*;
public class Solution{

    public static void main(String[] args)  {
        Solution s = new Solution();
    }

    /*输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
     **/
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> temp = new ArrayList<Integer>();
        ArrayList<Integer> result = new ArrayList<Integer>();
        if (listNode == null) return result;
        while (listNode != null) {
            temp.add(listNode.val);
            listNode = listNode.next;
        }
        for (int i = temp.size() - 1; i >= 0; i--) {
            result.add(temp.get(i));
        }
        return result;
    }

        /*
        输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
        例如
        输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
        * */
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        TreeNode tn = new TreeNode(pre[0]);
        int index = 0;
        for(int i = 0; i < in.length; i++){
            if(in[i] == pre[0]){
                index = i;
                break;
            }
        }
        if(pre.length == in.length && pre.length == 1){
            tn.right = null;
            tn.left = null;
            return tn;
        }
        if(index == 0)
        {
            tn.left = null;
            int[] subpre = new int[pre.length - 1];
            int[] subin = new int[in.length - 1];
            System.arraycopy(pre, 1, subpre, 0, subpre.length);
            System.arraycopy(in, 1, subin, 0, subin.length);
            tn.right = reConstructBinaryTree(subpre, subin);
        }
        else
            if (index == in.length - 1){
                int[] subpre = new int[pre.length - 1];
                int[] subin = new int[in.length - 1];
                System.arraycopy(pre, 1, subpre, 0, subpre.length);
                System.arraycopy(in, 0, subin, 0, subpre.length);
                tn.left = reConstructBinaryTree(subpre, subin);
                tn.right = null;
            }
            else {
                int[] subprel = new int[index];
                int[] subinl = new int[index];
                System.arraycopy(pre, 1, subprel, 0, subprel.length);
                System.arraycopy(in, 0, subinl, 0, subinl.length);
                tn.left = reConstructBinaryTree(subprel, subinl);

                int[] subprer = new int[in.length - (index + 1)];
                int[] subinr = new int[in.length - (index + 1)];
                System.arraycopy(pre, index + 1, subprer, 0, subprer.length);
                System.arraycopy(in, index + 1, subinr, 0, subinr.length);
                tn.right = reConstructBinaryTree(subprer, subinr);
            }

        return tn;

    }

    /*用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。*/
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
/*

    public void push(int node) {
        if (stack1.empty()){
            stack1.push(node);
        }
    }

    public int pop() {
        if (!stack2.empty())
            return stack2.pop();
        else {
            if (!stack1.empty()){
                while (!stack1.empty()){
                    stack2.push(stack1.pop());
                }
                return stack2.pop();
            }
            else
                return 0;
        }
    }
*/

    /*把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
    输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。
    例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
    * */

    public int minNumberInRotateArray(int [] array) {
        if (array.length == 0) return 0;
        if (array.length == 1) return array[0];
        int i = 0;
        while (i < array.length-2){
            if (array[i] > array[i + 1]) break;
            else i++;
        }
        return array[i + 1];
    }

    /*大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。n<=39
    * */

    public int Fibonacci(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        else return Fibonacci(n-1) + Fibonacci(n-2);

    }

    /*一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
    * */

    public int JumpFloor(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        if (target == 2) return 2;
        else return JumpFloor(target - 1) + JumpFloor(target - 2);
    }


    /*一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
    * */

    public int JumpFloorII(int target) {
        int sum = 0;
        if (target <= 0) return 0;
        else
            if (target == 1) return 1;
            else {
                while (target > 0){
                    sum += JumpFloorII(target-1);
                    target--;
                }
                return sum + 1;
            }
    }


    /*我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
    * */


    public int RectCover(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        if (target == 2) return 2;
        else
            return RectCover(target-1) + RectCover(target-2);
    }


    /*
    * 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。*/
    public int NumberOf1(int n) {
        int num = 1;
        int count = 0;
        while (num != 0){
            if ((n&num) != 0) count++;
            num = num << 1;
        }
        return count;
    }


    /*给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。*/

    public double Power(double base, int exponent) {
        if (base == 0 && exponent == 0) throw new RuntimeException("分子分母为0");
        double base2 = base;
        if (exponent == 0) return 1;
        boolean isPonum;
        if (exponent > 0) isPonum = true;
        else isPonum = false;
        if (!isPonum) exponent = -exponent;
        while (exponent != 1){
            base *= base2;
            exponent--;
        }
        return isPonum ? base:(1/base);
    }



    /*输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
    所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。*/

    public void reOrderArray(int [] array) {
        if (array.length == 0) return;
        int[] temp = new int[array.length];
        int odd = 0;
        int j = 0;
        int k = 0;
        for (int num:array
             ) {
            if (num %2 == 1)
                odd++;
        }
        for (int i = 0; i < array.length; i++){
            if (array[i] %2 ==1) {
                temp[j] = array[i];
                j++;
            }
            else{
                temp[k+odd] = array[i];
                k++;
            }
        }
        for (int i = 0; i < array.length; i++){
            array[i] = temp[i];
        }
    }


    /*输入一个链表，输出该链表中倒数第k个结点。*/

    public ListNode FindKthToTail(ListNode head,int k) {
        if (k <= 0 || head == null) return null;
        ListNode start = head;
        ListNode end = head;
        for (int i = 1; i < k; i++){
            if (start.next != null){
                start = start.next;
            }
            else return null;
        }
        while (start.next != null){
            start = start.next;
            end = end.next;
        }
        return end;

    }

    /*输入一个链表，反转链表后，输出新链表的表头。*/

    public ListNode ReverseList(ListNode head) {
        if (head == null) return null;
        ListNode pre = null;
        ListNode tmp = null;
        ListNode node = head;
        while (node.next != null){
            tmp = node.next;
            node.next = pre;
            pre = node;
            node = tmp;
        }
        node.next = pre;
        return node;
    }

    /*输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。*/

    public ListNode Merge(ListNode list1,ListNode list2) {
        if (list1 == null) return list2;
        if (list2 == null) return list1;

        int first = 0;
        if (list1.val < list2.val){
            first = list1.val;
            list1 = list1.next;

        }
        else
        {
            first = list2.val;
            list2 = list2.next;
        }
        ListNode head = new ListNode(first);
        ListNode node = head;
        ListNode tmp = null;
        while (list1 != null && list2 != null){
            if (list1.val < list2.val){
                node.next = list1;
                node = node.next;
                tmp = list1.next;
                list1.next = null;
                list1 = tmp;
            }
            else {
                node.next = list2;
                node = node.next;
                tmp = list2.next;
                list2.next = null;
                list2 = tmp;
            }
        }
        if (list1 != null){
            node.next = list1;
        }
        if (list2 != null){
            node.next = list2;
        }
        return head;
    }


    /*输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）*/

    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        if (root2 == null || root1 == null) return false;
        boolean result = false;
        if (root1.val == root2.val){
            result = IsSubtree(root1, root2);
        }
        if (!result)
            result = HasSubtree(root1.left, root2);
        if (!result)
            result = HasSubtree(root1.right, root2);
        return result;
    }
    public boolean IsSubtree(TreeNode root1, TreeNode root2){
        if (root2 == null) return true;
        else
            if (root1 == null) return false;
            else {
                if (root1.val == root2.val) return IsSubtree(root1.left, root2.left) && IsSubtree(root1.right, root2.right);
                else return false;
            }
    }




    /*操作给定的二叉树，将其变换为源二叉树的镜像。*/

    public void Mirror(TreeNode root) {
        if (root == null) return;
        if (root.left == null && root.right == null) return;
        else
            if (root.left != null && root.right  == null){
            Mirror(root.left);
            root.right = root.left;
            root.left = null;
        }
            else
                if (root.left == null && root.right  != null){
                    Mirror(root.right);
                    root.left = root.right;
                    root.right = null;
                }
                else {
                    Mirror(root.left);
                    Mirror(root.right);
                    TreeNode tmp = root.right;
                    root.right = root.left;
                    root.left = tmp;
                }
    }



    /*输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
    例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.*/

    public ArrayList<Integer> printMatrix(int [][] matrix) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        if (matrix == null) return result;
        int rows = matrix.length;
        int cols = matrix[0].length;
        for (int i = 0;i < cols; i++){
            result.add(matrix[0][i]);
        }
        for (int i = 1; i < rows; i++){
            result.add(matrix[i][cols-1]);
        }
        if (rows > 1){
            for (int i = cols - 2; i >= 0; i--){
                result.add(matrix[rows-1][i]);
            }
        }

        if (cols > 1){
            for (int i = rows - 2; i > 0; i--){
                result.add(matrix[i][0]);
            }
        }

        int[][] submatrix = null;
        int subrows = rows - 2;
        int subcols = cols - 2;
        if (subrows > 0 && subcols > 0)
        {
            submatrix = new int[subrows][subcols];
            for (int i = 0; i < subrows; i++){
                for (int j = 0; j < subcols; j++){
                    submatrix[i][j] = matrix[i+1][j+1];
                }
            }
        }
        result.addAll(printMatrix(submatrix));
        return result;
    }



    /*定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。##############巧妙
     */
    Stack<Integer> minStack = new Stack<Integer>();
    Stack<Integer> seqStack = new Stack<Integer>();
    public void push(int node) {
        minStack.push(node);
        if (seqStack.empty() || node <= seqStack.peek()) seqStack.push(node);
    }

    public void pop() {
        if (minStack.peek() == seqStack.peek()) seqStack.pop();
        minStack.pop();
    }

    public int top() {
        return minStack.peek();
    }

    public int min() {
        return seqStack.peek();
    }


    /*输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。##############巧妙
    假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，
    序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）*/


    public boolean IsPopOrder(int [] pushA,int [] popA) {
        if (pushA.length == 0 || popA.length == 0) return false;
        Stack<Integer> IspoporderS = new Stack<Integer>();
        int index = 0;
        for (int i = 0; i<pushA.length; i++){
            IspoporderS.push(pushA[i]);
            while (!IspoporderS.empty() && IspoporderS.peek() == popA[index]){
                IspoporderS.pop();
                index++;
            }
        }
        return IspoporderS.empty();
    }


    /*从上往下打印出二叉树的每个节点，同层节点从左至右打印。*/

    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> orderprint = new ArrayList<Integer>();
        Queue<TreeNode> PrintTreeQue = new LinkedList<TreeNode>();
        if (root == null) return orderprint;
        orderprint.add(root.val);
        while (root != null){
            if (root.left != null){
                orderprint.add(root.left.val);
                PrintTreeQue.offer(root.left);
            }
            if (root.right != null){
                orderprint.add(root.right.val);
                PrintTreeQue.offer(root.right);
            }
            root = PrintTreeQue.poll();
        }
        return orderprint;
    }


    /*输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
    如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。*/


    public boolean VerifySquenceOfBST(int [] sequence) {
        if (sequence.length == 0) return false;
        boolean result = VerifySquenceBST(sequence, 0, sequence.length-1);
        return result;
    }
    public boolean VerifySquenceBST(int [] sequence, int start, int end) {
        if (start > end) return true;
        int root = sequence[end];
        int mid = 0;
        boolean result = true;
        for (int i = start; i <= end; i++){
            if (sequence[i] > root){
                mid = i;
                break;
            }
            if (i == end) return true;
        }
        for (int i = mid; i <= end - 1; i++){
            if (sequence[i] < root){
                result = false;
                break;
            }
        }
        return result && VerifySquenceBST(sequence, start, mid -1) && VerifySquenceBST(sequence, mid, end-1);

    }


    /*输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。 ############
    路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)*/

    ArrayList<ArrayList<Integer>> paths = new ArrayList<ArrayList<Integer>>();
    ArrayList<Integer> path = new ArrayList<Integer>();
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
        if (root == null) return paths;
        path.add(root.val);
        target -= root.val;
        if (target == 0 && root.left == null && root.right == null){
            paths.add(new ArrayList<Integer>(path));
        }

        FindPath(root.left, target);
        FindPath(root.right, target);
        path.remove(path.size() - 1);
        return paths;

    }


    /*输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
    返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）*/

    public RandomListNode Clone(RandomListNode pHead)
    {
        if (pHead == null) return null;
        RandomListNode head = new RandomListNode(pHead.label);
        RandomListNode node = head;
        while (pHead != null){
            if (pHead.next != null){
                RandomListNode nodenext = new RandomListNode(pHead.next.label);
                node.next = nodenext;
            }
            if (pHead.random != null){
                RandomListNode noderandom = new RandomListNode(pHead.random.label);
                node.random = noderandom;
            }
            node = node.next;
            pHead = pHead.next;
        }
        return head;
    }


    /*输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
    * 此处用到了非递归中序遍历 需要仔细复习*/

    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) return null;
        TreeNode pre = null;
        TreeNode p = pRootOfTree;
        boolean isFirst = true;
        Stack<TreeNode> stack = new Stack<TreeNode>();
        while (p != null || !stack.empty()){
            while (p != null){
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (isFirst){
                pRootOfTree = p;
                pre = p;
                isFirst = false;
            }
            else {
                pre.right = p;
                p.left = pre;
                pre = p;
            }
            p = p.right;
        }
        return pRootOfTree;
    }


    /*输入一个字符串,按字典序打印出该字符串中字符的所有排列。
    例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。*/

//这一段就是回溯法，这里以"abc"为例

    //递归的思想与栈的入栈和出栈是一样的,某一个状态遇到return结束了之后，会回到被调用的地方继续执行

    //1.第一次进到这里是ch=['a','b','c'],list=[],i=0，我称为 状态A ，即初始状态
    //那么j=0，swap(ch,0,0)，就是['a','b','c']，进入递归，自己调自己，只是i为1，交换(0,0)位置之后的状态我称为 状态B
    //i不等于2，来到这里，j=1，执行第一个swap(ch,1,1)，这个状态我称为 状态C1 ,再进入fun函数，此时标记为T1，i为2，那么这时就进入上一个if，将"abc"放进list中
    /////////////-------》此时结果集为["abc"]

    //2.执行完list.add之后，遇到return，回退到T1处，接下来执行第二个swap(ch,1,1)，状态C1又恢复为状态B
    //恢复完之后，继续执行for循环，此时j=2,那么swap(ch,1,2),得到"acb"，这个状态我称为C2,然后执行fun，此时标记为T2,发现i+1=2,所以也被添加进结果集，此时return回退到T2处往下执行
    /////////////-------》此时结果集为["abc","acb"]
    //然后执行第二个swap(ch,1,2)，状态C2回归状态B,然后状态B的for循环退出回到状态A

    //             a|b|c(状态A)
    //               |
    //               |swap(0,0)
    //               |
    //             a|b|c(状态B)
    //             /  \
    //   swap(1,1)/    \swap(1,2)  (状态C1和状态C2)
    //           /      \
    //         a|b|c   a|c|b

    //3.回到状态A之后，继续for循环，j=1,即swap(ch,0,1)，即"bac",这个状态可以再次叫做状态A,下面的步骤同上
    /////////////-------》此时结果集为["abc","acb","bac","bca"]

    //             a|b|c(状态A)
    //               |
    //               |swap(0,1)
    //               |
    //             b|a|c(状态B)
    //             /  \
    //   swap(1,1)/    \swap(1,2)  (状态C1和状态C2)
    //           /      \
    //         b|a|c   b|c|a

    //4.再继续for循环，j=2,即swap(ch,0,2)，即"cab",这个状态可以再次叫做状态A，下面的步骤同上
    /////////////-------》此时结果集为["abc","acb","bac","bca","cab","cba"]

    //             a|b|c(状态A)
    //               |
    //               |swap(0,2)
    //               |
    //             c|b|a(状态B)
    //             /  \
    //   swap(1,1)/    \swap(1,2)  (状态C1和状态C2)
    //           /      \
    //         c|b|a   c|a|b

    //5.最后退出for循环，结束。
    //回溯法在此处的思想即在1次循环中确保前1个数的位置不变，通过递归来保证前n-1个数不变，每退出一层递归则交换后面的字符
    //故当一次循环完成时,以第一个字符为开头的所有组合以添加完毕，而后将第一个字符换到第2 3 4 5个位置重复上述过程
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> strs = new ArrayList<String>();
        if (str.length() == 0) return strs;
        fun(str.toCharArray(), strs, 0);
        Collections.sort(strs);
        return strs;
    }
    public void fun(char[] str, ArrayList<String> arrlist, int i){
        if (i == str.length-1){
            if (!arrlist.contains(new String(str))){
                arrlist.add(new String(str));
            }
        }
        else {
            for (int j = i; j < str.length; j++){
                swap(str, i, j);
                fun(str, arrlist, i+1);
                swap(str, i, j);
            }
        }
    }
    public void swap(char[] str, int i, int j){
        if(i != j){
            char tmp = str[j];
            str[j] = str[i];
            str[i] = tmp;
        }
    }



    /*数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
    例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。*/

    public int MoreThanHalfNum_Solution(int [] array) {
        HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
        for (int i = 0; i < array.length; i++){
            if (hm.containsKey(array[i])){
                int value = hm.get(array[i]);
                hm.put(array[i], ++value);
            }
            else hm.put(array[i], 1);
        }
        for (int i = 0; i < array.length; i++){
            if (hm.get(array[i]) > (array.length/2)) return array[i];
        }
        return 0;
    }

    /*输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
    * */

    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        boolean flag;
        if (k <= 0 || k > input.length) return result;
        if (k == input.length){
            for (int i = 0; i < input.length; i++){
                result.add(input[i]);
            }
            return result;
        }
        for (int i = 0; i < input.length; i++) {
            flag = false;
            for (int j = input.length - 1; j > i; j--) {
                if (input[j] < input[j - 1]){
                    int tmp = input[j - 1];
                    input[j - 1] = input[j];
                    input[j] = tmp;
                    flag = true;
                }
            }
            if (flag == false || i > k) break;
        }
        for (int i = 0; i < k; i++) {
            result.add(input[i]);
        }
        return result;
    }


    /*HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:
    在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。
    但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？
    例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，
    你会不会被他忽悠住？(子向量的长度至少是1)*/

    /*
使用动态规划
F（i）：以array[i]为末尾元素的子数组的和的最大值，子数组的元素的相对位置不变
F（i）=max（F（i-1）+array[i] ， array[i]）
res：所有子数组的和的最大值
res=max（res，F（i））

如数组[6, -3, -2, 7, -15, 1, 2, 2]
初始状态：
    F（0）=6
    res=6
i=1：
    F（1）=max（F（0）-3，-3）=max（6-3，3）=3
    res=max（F（1），res）=max（3，6）=6
i=2：
    F（2）=max（F（1）-2，-2）=max（3-2，-2）=1
    res=max（F（2），res）=max（1，6）=6
i=3：
    F（3）=max（F（2）+7，7）=max（1+7，7）=8
    res=max（F（2），res）=max（8，6）=8
i=4：
    F（4）=max（F（3）-15，-15）=max（8-15，-15）=-7
    res=max（F（4），res）=max（-7，8）=8
以此类推
最终res的值为8*/

    public int FindGreatestSumOfSubArray(int[] array) {
        int[] maxi = new int[array.length];
        maxi[0] = array[0]; //记录以第i个元素为结尾的子数组和的最大值
        int sum = array[0];
        for (int i = 1; i < maxi.length; i++) {
            maxi[i] = max(maxi[i - 1] + array[i], array[i]);
            if (maxi[i] > sum) sum = maxi[i];
        }
        return sum;
    }
    public int max(int i, int j){
        if (i > j) return i;
        else return j;
    }


    /*求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
    为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。
    ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。*/

    public int NumberOf1Between1AndN_Solution(int n) {
        if (n < 1) return 0;
        //主要思路：设定整数点（如1、10、100等等）作为位置点i（对应n的各位、十位、百位等等），分别对每个数位上有多少包含1的点进行分析
        //根据设定的整数位置，对n进行分割，分为两部分，高位n/i，低位n%i
        //当i表示百位，且百位对应的数>=2,如n=31456,i=100，则a=314,b=56，此时百位为1的次数有a/10+1=32（最高两位0~31），每一次都包含100个连续的点，即共有(a%10+1)*100个点的百位为1
        //当i表示百位，且百位对应的数为1，如n=31156,i=100，则a=311,b=56，此时百位对应的就是1，则共有a%10(最高两位0-30)次是包含100个连续点，当最高两位为31（即a=311），本次只对应局部点00~56，共b+1次，所有点加起来共有（a%10*100）+(b+1)，这些点百位对应为1
        //当i表示百位，且百位对应的数为0,如n=31056,i=100，则a=310,b=56，此时百位为1的次数有a/10=31（最高两位0~30）
        //综合以上三种情况，当百位对应0或>=2时，有(a+8)/10次包含所有100个点，还有当百位为1(a%10==1)，需要增加局部点b+1
        //之所以补8，是因为当百位为0，则a/10==(a+8)/10，当百位>=2，补8会产生进位位，效果等同于(a/10+1)
        int cnt = 0;
        for (int m = 1; m <= n; m *= 10) {
            int a = n / m, b = n % m;
            cnt += (a + 8) / 10 * m + (a % 10 == 1 ? b + 1 : 0);
        }
        return cnt;
    }


    /*输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
    例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。*/

    public String PrintMinNumber(int [] numbers) {
        String result = new String();
        ArrayList<Integer> list = new ArrayList<>();
        if (numbers.length == 0) return result;
        for (int i = 0; i < numbers.length; i++) {
            list.add(numbers[i]);
        }
        Collections.sort(list, new Comparator<Integer>() {
            @Override
            public int compare(Integer i1, Integer i2) {
                String str1 = i1 + "" + i2;
                String str2 = i2 + "" + i1;
                return str1.compareTo(str2);
            }
        });
        for (int i = 0; i < list.size(); i++) {
            result +=  list.get(i);
        }
        return result;
    }


    /*把只包含质因子2、3和5的数称作丑数（Ugly Number）。
    例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。*/


    /*此题有两点需要注意
    * 对于任何丑数p：
（一）那么2*p,3*p,5*p都是丑数，并且2*p<3*p<5*p 也就是通过这个公式去获取丑数
（二）如果p<q, 那么2*p<2*q,3*p<3*q,5*p<5*q  通过这个公式说明新生成的较大的丑数有两个是不用比较的   */
    public int GetUglyNumber_Solution(int index) {
        if (index <= 0) return 0;
        int[] unlyre = new int[index];
        unlyre[0] = 1;
        int t2 = 0, t3 = 0, t5 = 0; //标记选取乘以哪个数作为新生成的丑数
        for (int i = 1; i < unlyre.length; i++) {
            int tmp = Math.min(unlyre[t2] * 2, unlyre[t3] * 3);
            unlyre[i] = Math.min(tmp,  unlyre[t5] * 5);
            if (unlyre[i] == unlyre[t2] * 2) t2++; //代表是乘以2的数生成的丑数该位后移
            if (unlyre[i] == unlyre[t3] * 3) t3++;
            if (unlyre[i] == unlyre[t5] * 5) t5++;
        }
        return unlyre[index - 1];
    }


    /*在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置,
    如果没有则返回 -1（需要区分大小写）.*/

    public int FirstNotRepeatingChar(String str) {
        if (str.length() == 0) return -1;
        char[] chars = str.toCharArray();
        HashMap<Character, Integer> hm = new HashMap<Character, Integer>();
        for (int i = 0; i < chars.length; i++) {
            if (!hm.containsKey(chars[i])){
                hm.put(chars[i], i);
            }
            else hm.put(chars[i], -1);
        }
        ArrayList<Integer> arr = new ArrayList<Integer>(hm.values());
        int min = str.length();
        for (int i: arr
             ) {
            if (i > -1 && i < min) min = i;
        }
        return min;
    }


    /*在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
    输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007*/

    public int InversePairs(int [] array) {
        if (array.length < 2) return 0;
        int[] copy = new int[array.length];
        for (int i = 0; i < copy.length; i++) {
            copy[i] = array[i];

        }
        return InverseCount(array, copy, 0, array.length-1);
    }
    public int InverseCount(int[] array, int[] copy, int start, int end){
        if (start >= end){
            return 0;
        }
        int mid = (start + end) >> 1;
        int leftcount = InverseCount(array, copy, start, mid) % 1000000007;
        int rightcount = InverseCount(array, copy, mid + 1, end) % 1000000007;
        int count = 0;
        int left = mid, right = end, k = end;
        while (left >= start && right > mid){
            if (array[left] <= array[right]){
                copy[k--] = array[right--];
            }
            else {
                copy[k--] = array[left--];
                count += (right - mid);
                if(count>=1000000007)// 必须在此处进行判断！！！！！！！否则无法处理过多的数据
                {
                    count%=1000000007;
                }
            }
        }
        for (; left >= start; left--){
            copy[k--] = array[left];
        }
        for (; right > mid; right--){
            copy[k--] = array[right];
        }
        for (int i = start; i <= end; i++) {
            array[i] = copy[i];
        }
        return (count+leftcount+rightcount) % 1000000007;
    }

    /*
    * 输入两个链表，找出它们的第一个公共结点。*/

    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode p1 = pHead1;
        ListNode p2 = pHead2;
        if (p1 == null || p2 == null) return null;
        while (p1 != p2){
            p1 = p1 == null?  pHead2 : p1.next;
            p2 = p2 == null?  pHead1 : p2.next;
        }
        return p1;
    }

    /*统计一个数字在排序数组中出现的次数。*/

    public int GetNumberOfK(int [] array , int k) {
            int count = 0;
            if (array.length == 0) return 0;
            if (array.length == 1) return array[0] ==  k ? 1:0;
            boolean isAs = (array[0] <= array[1]);
            if (isAs){
                for (int i = 0; i < array.length; i++) {
                    if (array[i] == k) count++;
                    if (array[i] > k) break;
                }
            }
            else {
                for (int i = 0; i < array.length; i++) {
                    if (array[i] == k) count++;
                    if (array[i] < k) break;
                }
            }
            return count;
    }

    /*输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，
    最长路径的长度为树的深度。*/

    public int TreeDepth(TreeNode root) {
        return 1;
    }
}
